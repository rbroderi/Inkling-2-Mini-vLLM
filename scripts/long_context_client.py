"""Client of long_context_probe.sh: one synthetic ~90% prompt against a running server.

Prints one tab-separated line: prefill_s, prompt_tokens, decode_tok_s, needle (pass|fail). Needs only the standard
library.
"""

import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8011)
ap.add_argument("--length", type=int, required=True)
ap.add_argument("--decode-tokens", type=int, default=300)
a = ap.parse_args()
base = f"http://localhost:{a.port}"
CODE = "7391-AQUA-4408"


def post(path, body, timeout=3600):
    """POST a JSON body to the local server and return the open response."""
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})  # noqa: S310  # local http server
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # local http server


def ntok(text):
    """Return the number of tokens the server counts in ``text``."""
    body = {"model": "inkling-2-mini", "prompt": text, "add_special_tokens": False}
    return len(json.load(post("/tokenize", body))["tokens"])


def sentence(i):
    """Return the i-th filler sentence."""
    return (
        f"Note {i}: shelf {i % 97} of archive {i % 13} holds item {(i * 7919) % 10007} "
        f"and was checked on day {i % 365}.\n"
    )


target = int(a.length * 0.9) - 200
per = ntok("".join(sentence(i) for i in range(2000))) / 2000
n = int(target / per)
lines = [sentence(i) for i in range(n)]
lines.insert(n // 2, f"IMPORTANT: the secret passcode for the vault is {CODE}. Remember it.\n")
haystack = "".join(lines)
tokens = ntok(haystack)
question = "\n\nWhat is the secret passcode for the vault? Answer with the passcode only."
chat = {"model": "inkling-2-mini", "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"}}


def msg(tail):
    """Return the chat messages: the haystack followed by ``tail``."""
    return [{"role": "user", "content": haystack + tail}]


t0 = time.time()  # prefill: a cold prompt, one output token
post("/v1/chat/completions", {**chat, "messages": msg(question), "max_tokens": 1}).read()
prefill = time.time() - t0

r = json.load(post("/v1/chat/completions", {**chat, "messages": msg(question), "max_tokens": 40}))
needle = "pass" if CODE in r["choices"][0]["message"]["content"] else "fail"

# Decode speed at this context (the shared prefix is cached, so this is almost pure decode).
body = {
    **chat,
    "messages": msg("\n\nWrite a long story about the archive."),
    "max_tokens": a.decode_tokens,
    "min_tokens": a.decode_tokens,
    "stream": True,
    "stream_options": {"include_usage": True},
}
first = last = None
count = 0
for raw in post("/v1/chat/completions", body):
    line = raw.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    d = json.loads(line[6:])
    if d.get("usage"):
        count = d["usage"]["completion_tokens"]
    elif d["choices"] and d["choices"][0]["delta"].get("content"):
        now = time.time()
        first = first or now
        last = now
rate = (count - 1) / (last - first) if first and last and last > first else 0.0
print(f"{prefill:.1f}\t{tokens}\t{rate:.1f}\t{needle}")
