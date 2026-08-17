"""
Mock Cloud — fake Bedrock, Vertex, Foundry, and Direct API endpoints

Runs every lab in this repo offline, with no cloud account and no credentials.
The lab scripts are not modified: the Anthropic and Azure SDKs each read their
base URL from the environment, so pointing them here is a .env change.

Usage:
  python mock_cloud/server.py                 # serves on 127.0.0.1:8787
  python mock_cloud/server.py --port 9000
  python mock_cloud/server.py --selftest      # assert the reply logic, exit

Then in .env:
  ANTHROPIC_BASE_URL=http://127.0.0.1:8787/direct
  ANTHROPIC_BEDROCK_BASE_URL=http://127.0.0.1:8787/bedrock
  ANTHROPIC_VERTEX_BASE_URL=http://127.0.0.1:8787/vertex/v1
  AZURE_FOUNDRY_ENDPOINT=http://127.0.0.1:8787/foundry

Credentials still have to be *present* for the SDKs to construct a client, but
they are never checked — see mock_cloud/README.md for the dummy values.

What this is: real HTTP, real SSE, real AWS event-stream framing, real tool_use
blocks, real multi-turn agentic loops. Everything except a real model.
What this is not: a model. Responses are canned. It cannot answer your prompt.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Fake per-platform latency so Lab 04's comparison table looks like a real run.
PLATFORM_DELAY = {"direct": 0.25, "bedrock": 0.55, "vertex": 0.70, "foundry": 0.60}

CITIES = ["San Francisco", "New York", "Seattle"]


# --------------------------------------------------------------------------- #
#  Canned text                                                                  #
# --------------------------------------------------------------------------- #

CANNED = {
    "foundation model": (
        "A foundation model is a large neural network trained on a broad corpus of data, "
        "which can then be adapted to many downstream tasks rather than built for just one. "
        "For a software engineer, the useful analogy is a general-purpose library you call "
        "through an API instead of a single-purpose function you write yourself."
    ),
    "attention": (
        "Attention lets every token in a sequence look at every other token and decide which "
        "ones matter for interpreting it. Each token is projected into three vectors — a query, "
        "a key, and a value. The query of one token is compared against the keys of all tokens "
        "to produce a set of scores, those scores are normalised with a softmax into weights, "
        "and the output for that position is the weighted sum of the value vectors. Multi-head "
        "attention runs several of these in parallel with different projections, so one head can "
        "track syntax while another tracks long-range reference. Because every position is "
        "computed independently of the others, the whole operation is a couple of matrix "
        "multiplications, which is what makes transformers train efficiently on modern hardware."
    ),
    "multiple cloud": (
        "Enterprises deploy AI across several clouds to avoid concentrating negotiating leverage "
        "and operational risk in one vendor. Different platforms also lead on different things at "
        "different times — region availability, compliance certifications, committed-spend pricing "
        "— and a portable integration layer lets a team take whichever is currently best. The cost "
        "is that every platform has its own SDK, quotas, and failure modes to learn."
    ),
    "chain-of-thought": (
        "Chain-of-thought prompting asks a model to write out its intermediate reasoning before "
        "committing to an answer, rather than emitting the answer directly. On multi-step problems "
        "this measurably improves accuracy, because each step conditions the next."
    ),
}

GENERIC = (
    "This is a canned response from the mock cloud server. No model was involved, so the text "
    "does not actually address the prompt — it exists so the lab's request path, streaming, "
    "token accounting, and tool loop can be exercised offline."
)


def canned_text(prompt: str) -> str:
    low = prompt.lower()
    for key, text in CANNED.items():
        if key in low:
            return text
    return GENERIC


def fake_tokens(text: str) -> int:
    """Rough token estimate. Real tokenizers differ; this only needs to be plausible."""
    return max(1, int(len(text.split()) * 1.3))


# --------------------------------------------------------------------------- #
#  Conversation inspection                                                      #
# --------------------------------------------------------------------------- #
#  The API is stateless, so the full history arrives on every request. That is
#  enough to work out what the "model" should do next without the server holding
#  any state of its own.

def _blocks(msg: dict) -> list:
    content = msg.get("content")
    return content if isinstance(content, list) else []


def prior_tool_uses(messages: list[dict]) -> list[dict]:
    """Every tool_use block the assistant has already emitted, oldest first."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for block in _blocks(m):
            if block.get("type") == "tool_use":
                out.append(block)
    return out


def first_user_text(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            for b in _blocks(m):
                if isinstance(b, dict) and b.get("type") == "text":
                    return b.get("text", "")
    return ""


def _tool_use(name: str, args: dict) -> dict:
    return {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:16]}", "name": name, "input": args}


def decide_tool_calls(messages: list[dict], tool_names: set[str]) -> list[dict]:
    """Return the tool_use blocks to emit this turn, or [] to finish with text."""
    used = prior_tool_uses(messages)
    prompt = first_user_text(messages)

    # --- Lab 06: workflow tools. Plan once, then one finding per turn. ---
    if "make_plan" in tool_names:
        plans = [u for u in used if u["name"] == "make_plan"]
        if not plans:
            topic = prompt.replace("Research the topic:", "").strip() or "the topic"
            return [_tool_use("make_plan", {"steps": [
                f"What is {topic} and how is it defined?",
                f"How does {topic} work in practice?",
                f"Where is {topic} applied today?",
            ]})]

        steps = plans[-1]["input"].get("steps", [])
        recorded = {u["input"].get("step") for u in used if u["name"] == "record_finding"}
        for step in steps:
            if step not in recorded:
                return [_tool_use("record_finding", {
                    "step": step,
                    "content": (
                        f"Canned finding for {step!r}. The mock server returns fixed text so the "
                        "plan-execute-synthesize loop can run without a model."
                    ),
                })]
        return []  # every step has a finding — synthesize

    # --- Labs 03 and 05: capability tools, all called on the first turn. ---
    if used:
        return []  # results are already back; finish with text

    calls = []
    if "get_weather" in tool_names:
        found = [c for c in CITIES if c.lower() in prompt.lower()] or [CITIES[0]]
        calls += [_tool_use("get_weather", {"city": c}) for c in found]
    if "search_docs" in tool_names:
        topic = "MCP" if "mcp" in prompt.lower() else prompt.split()[0] if prompt else "Claude"
        calls.append(_tool_use("search_docs", {"query": topic}))
    if "run_calculation" in tool_names:
        m = re.search(r"(\d+\s*[-+*/]\s*\d+)", prompt)
        if m:
            calls.append(_tool_use("run_calculation", {"expression": m.group(1)}))

    return calls


def final_text(messages: list[dict], tool_names: set[str]) -> str:
    used = prior_tool_uses(messages)
    if "make_plan" in tool_names and used:
        findings = [u for u in used if u["name"] == "record_finding"]
        lines = "\n".join(f"- {f['input'].get('step')}" for f in findings)
        return (
            f"Research complete. {len(findings)} findings recorded across the plan:\n\n{lines}\n\n"
            "This report is canned output from the mock cloud server — the structure is real, "
            "the content is not."
        )
    if used:
        names = ", ".join(sorted({u["name"] for u in used}))
        return (
            f"Based on the results from {names}: this is a canned synthesis from the mock cloud "
            "server. The tool loop ran for real — the model did not."
        )
    return canned_text(first_user_text(messages))


def build_reply(body: dict) -> dict:
    """Produce an Anthropic-shaped message response for a request body."""
    messages = body.get("messages", [])
    tool_names = {t.get("name") for t in (body.get("tools") or [])}

    calls = decide_tool_calls(messages, tool_names) if tool_names else []
    if calls:
        content, stop_reason = calls, "tool_use"
    else:
        content = [{"type": "text", "text": final_text(messages, tool_names)}]
        stop_reason = "end_turn"

    rendered = json.dumps(content)
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "claude-mock"),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": fake_tokens(json.dumps(messages)),
            "output_tokens": fake_tokens(rendered),
        },
    }


# --------------------------------------------------------------------------- #
#  Anthropic SSE event stream                                                   #
# --------------------------------------------------------------------------- #

def sse_events(reply: dict) -> list[tuple[str, dict]]:
    """Expand a finished reply into the event sequence a streaming call expects."""
    head = {k: v for k, v in reply.items() if k != "content"}
    head["content"] = []
    events: list[tuple[str, dict]] = [("message_start", {"type": "message_start", "message": head})]

    for i, block in enumerate(reply["content"]):
        if block["type"] == "text":
            events.append(("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""},
            }))
            # Chunk on word boundaries so the terminal output looks like generation.
            words = block["text"].split(" ")
            for n, w in enumerate(words):
                events.append(("content_block_delta", {
                    "type": "content_block_delta", "index": i,
                    "delta": {"type": "text_delta", "text": w if n == 0 else " " + w},
                }))
        else:
            events.append(("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}},
            }))
            events.append(("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
            }))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))

    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reply["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": reply["usage"]["output_tokens"]},
    }))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


# --------------------------------------------------------------------------- #
#  AWS event-stream framing (Bedrock streaming)                                 #
# --------------------------------------------------------------------------- #
#  Bedrock does not return SSE. invoke-with-response-stream returns binary
#  vnd.amazon.eventstream frames, which the Anthropic SDK decodes with botocore.
#  Frame layout:
#    u32 total_len | u32 headers_len | u32 prelude_crc | headers | payload | u32 crc

def _header(name: str, value: str) -> bytes:
    n, v = name.encode(), value.encode()
    return struct.pack("!B", len(n)) + n + b"\x07" + struct.pack("!H", len(v)) + v


def aws_frame(payload: bytes) -> bytes:
    headers = (
        _header(":message-type", "event")
        + _header(":event-type", "chunk")
        + _header(":content-type", "application/json")
    )
    total = 16 + len(headers) + len(payload)
    prelude = struct.pack("!II", total, len(headers))
    prelude += struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF)
    body = prelude + headers + payload
    return body + struct.pack("!I", zlib.crc32(body) & 0xFFFFFFFF)


def bedrock_stream_frames(reply: dict) -> list[bytes]:
    frames = []
    for _, data in sse_events(reply):
        inner = json.dumps(data).encode()
        # The chunk shape carries the real event as a base64 blob.
        frames.append(aws_frame(json.dumps({"bytes": base64.b64encode(inner).decode()}).encode()))
    return frames


# --------------------------------------------------------------------------- #
#  OpenAI shape (Foundry)                                                       #
# --------------------------------------------------------------------------- #

def openai_usage(reply: dict) -> dict:
    i, o = reply["usage"]["input_tokens"], reply["usage"]["output_tokens"]
    return {"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o}


def to_openai(reply: dict) -> dict:
    text = "".join(b["text"] for b in reply["content"] if b["type"] == "text")
    tool_calls = [
        {
            "id": b["id"], "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
        }
        for b in reply["content"] if b["type"] == "tool_use"
    ]
    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": reply["id"], "object": "chat.completion", "created": int(time.time()),
        "model": reply["model"],
        "choices": [{
            "index": 0, "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": openai_usage(reply),
    }


def from_openai_messages(body: dict) -> dict:
    """Translate an OpenAI-shaped request into the Anthropic shape build_reply expects."""
    messages = []
    for m in body.get("messages", []):
        role = m.get("role")
        if role == "system":
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{"type": "tool_result", "content": m.get("content")}]})
            continue
        if role == "assistant" and m.get("tool_calls"):
            messages.append({"role": "assistant", "content": [
                {
                    "type": "tool_use", "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"] or "{}"),
                }
                for tc in m["tool_calls"]
            ]})
            continue
        messages.append({"role": role, "content": m.get("content") or ""})

    tools = [
        {"name": t["function"]["name"]}
        for t in (body.get("tools") or [])
        if isinstance(t, dict) and "function" in t
    ]
    return {"messages": messages, "tools": tools, "model": body.get("model", "claude-mock")}


def openai_chunks(reply: dict) -> list[dict]:
    base = {"id": reply["id"], "object": "chat.completion.chunk", "created": int(time.time()),
            "model": reply["model"]}
    out = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
    for block in reply["content"]:
        if block["type"] != "text":
            continue
        words = block["text"].split(" ")
        for n, w in enumerate(words):
            out.append({**base, "choices": [
                {"index": 0, "delta": {"content": w if n == 0 else " " + w}, "finish_reason": None}
            ]})
    out.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": openai_usage(reply)})
    return out


# --------------------------------------------------------------------------- #
#  HTTP                                                                         #
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockCloud/1.0"

    def log_message(self, fmt, *args):
        print(f"[mock] {self.address_string()} {fmt % args}")

    # -- response helpers --

    def _json(self, obj: dict, status: int = 200):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _begin_chunked(self, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # -- routing --

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"type": "error", "error": {"type": "invalid_request_error",
                                                          "message": "malformed JSON"}}, 400)

        path = self.path.split("?")[0]
        platform = next((p for p in PLATFORM_DELAY if f"/{p}" in path), "direct")
        time.sleep(PLATFORM_DELAY[platform])  # simulated network latency

        try:
            if path.endswith("/chat/completions"):
                return self._foundry(body)
            if path.endswith("invoke-with-response-stream"):
                return self._bedrock_stream(body)
            if path.endswith("/invoke") or path.endswith(":rawPredict"):
                return self._json(build_reply(body))
            if path.endswith(":streamRawPredict") or path.endswith("/v1/messages"):
                if body.get("stream"):
                    return self._sse(build_reply(body))
                return self._json(build_reply(body))
        except Exception as exc:  # a mock that 500s silently is worse than useless
            self.log_message("handler error: %r", exc)
            return self._json({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, 500)

        self._json({"type": "error", "error": {"type": "not_found_error",
                                               "message": f"no mock route for {path}"}}, 404)

    # -- per-platform responses --

    def _sse(self, reply: dict):
        self._begin_chunked("text/event-stream")
        for name, data in sse_events(reply):
            self._chunk(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        self._end_chunked()

    def _bedrock_stream(self, body: dict):
        self._begin_chunked("application/vnd.amazon.eventstream")
        for frame in bedrock_stream_frames(build_reply(body)):
            self._chunk(frame)
        self._end_chunked()

    def _foundry(self, body: dict):
        reply = build_reply(from_openai_messages(body))
        if not body.get("stream"):
            return self._json(to_openai(reply))
        self._begin_chunked("text/event-stream")
        for chunk in openai_chunks(reply):
            self._chunk(f"data: {json.dumps(chunk)}\n\n".encode())
        self._chunk(b"data: [DONE]\n\n")
        self._end_chunked()


# --------------------------------------------------------------------------- #
#  Self-check                                                                   #
# --------------------------------------------------------------------------- #

def selftest():
    # Plain text, no tools.
    r = build_reply({"messages": [{"role": "user", "content": "What is a foundation model?"}]})
    assert r["stop_reason"] == "end_turn", r["stop_reason"]
    assert "foundation model" in r["content"][0]["text"]

    # Lab 03: two cities in the prompt must produce two tool_use blocks.
    weather = {"name": "get_weather"}
    turn1 = build_reply({
        "messages": [{"role": "user", "content": "Weather in San Francisco and Seattle?"}],
        "tools": [weather],
    })
    assert turn1["stop_reason"] == "tool_use"
    assert [b["input"]["city"] for b in turn1["content"]] == ["San Francisco", "Seattle"]

    # ...and once results are back, it must finish rather than loop.
    turn2 = build_reply({
        "messages": [
            {"role": "user", "content": "Weather in San Francisco and Seattle?"},
            {"role": "assistant", "content": turn1["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "{}"}]},
        ],
        "tools": [weather],
    })
    assert turn2["stop_reason"] == "end_turn", turn2["stop_reason"]

    # Lab 06: plan first, then one finding per turn, then synthesize.
    tools = [{"name": n} for n in ("make_plan", "record_finding", "get_state")]
    msgs = [{"role": "user", "content": "Research the topic: quantum computing"}]
    first = build_reply({"messages": msgs, "tools": tools})
    assert first["content"][0]["name"] == "make_plan"
    steps = first["content"][0]["input"]["steps"]

    msgs.append({"role": "assistant", "content": first["content"]})
    for expected in steps:
        nxt = build_reply({"messages": msgs, "tools": tools})
        assert nxt["content"][0]["name"] == "record_finding", nxt["content"][0]["name"]
        assert nxt["content"][0]["input"]["step"] == expected
        msgs.append({"role": "assistant", "content": nxt["content"]})
    done = build_reply({"messages": msgs, "tools": tools})
    assert done["stop_reason"] == "end_turn", "agent must terminate once every step has a finding"

    # Bedrock frames must decode with the same library the SDK uses.
    from botocore.eventstream import EventStreamBuffer
    buf = EventStreamBuffer()
    for f in bedrock_stream_frames(r):
        buf.add_data(f)
    decoded = [json.loads(base64.b64decode(json.loads(e.payload)["bytes"])) for e in buf]
    assert decoded[0]["type"] == "message_start"
    assert decoded[-1]["type"] == "message_stop"

    # Foundry round trip: OpenAI request in, OpenAI response out.
    oa = to_openai(build_reply(from_openai_messages({
        "messages": [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "What is a foundation model?"}],
    })))
    assert oa["choices"][0]["finish_reason"] == "stop"
    assert oa["usage"]["total_tokens"] > 0

    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-delay", action="store_true", help="disable simulated latency")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.no_delay:
        for k in PLATFORM_DELAY:
            PLATFORM_DELAY[k] = 0.0

    print(f"[mock] serving Bedrock, Vertex, Foundry and Direct API on http://127.0.0.1:{args.port}")
    print("[mock] responses are canned — no model is involved")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
