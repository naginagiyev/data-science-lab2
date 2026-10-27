import os
import sys
import json
import time
import argparse
import threading
import urllib.error
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://api.openai.com"
ALLOWED_PREFIXES = ("/v1/chat/completions", "/v1/responses", "/v1/models")

# the token counts of one OpenAI usage record (chat completions or responses API)
def usage_tokens(usage):
    if not usage:
        return 0, 0
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return int(input_tokens), int(output_tokens)

# the usage record inside one server-sent event line of a streamed answer, or None
def stream_usage(line):
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:") or text == "data: [DONE]":
        return None
    try:
        event = json.loads(text[5:])
    except json.JSONDecodeError:
        return None
    return event.get("usage") or (event.get("response") or {}).get("usage")

# the shared state of the proxy: the allowed models, prices, the dollar cap of...
# the cell, the running cost and the usage log
class Ledger:

    # stores the settings and opens the usage log
    def __init__(self, models, price_input, price_output, cap_usd, log_file):
        self.models = set(models)
        self.price_input = price_input
        self.price_output = price_output
        self.cap_usd = cap_usd
        self.log_file = Path(log_file)
        self.cost = 0.0
        self.lock = threading.Lock()

    # true while the cell is still under its dollar cap
    def open(self):
        with self.lock:
            return self.cost < self.cap_usd

    # adds one call to the running cost and writes it as one JSON line
    def add(self, path, model, input_tokens, output_tokens, status, seconds):
        cost = (input_tokens * self.price_input + output_tokens * self.price_output) / 1e6
        record = {
            "time": time.time(), "path": path, "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost, "status": status, "seconds": round(seconds, 3),
        }
        with self.lock:
            self.cost += cost
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

# forwards one request to OpenAI with the real key, refuses other models and...
# calls over the cap, and records the tokens of every answer (also streamed ones)
class ProxyHandler(BaseHTTPRequestHandler):
    ledger = None
    api_key = ""

    # sends a JSON error answer in the OpenAI error format
    def refuse(self, status, message):
        body = json.dumps({"error": {"message": message, "type": "proxy_refused", "code": status}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # opens the upstream request with the real key
    def upstream(self, method, body=None):
        request = urllib.request.Request(UPSTREAM + self.path, data=body, method=method)
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(request, timeout=600)

    # model lists are forwarded without cost
    def do_GET(self):
        if not self.path.startswith("/v1/models"):
            return self.refuse(404, f"path {self.path} is not allowed")
        try:
            with self.upstream("GET") as response:
                body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as error:
            self.refuse(error.code, error.read().decode("utf-8", errors="replace")[:500])

    # chat and responses calls: checks the model and the cap, forwards the call...
    # and records its usage
    def do_POST(self):
        started = time.perf_counter()
        if not self.path.startswith(ALLOWED_PREFIXES):
            return self.refuse(404, f"path {self.path} is not allowed")
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        model = payload.get("model", "")
        if model not in self.ledger.models:
            self.ledger.add(self.path, model, 0, 0, "refused-model", 0.0)
            return self.refuse(400, f"model {model} is not allowed in this benchmark (allowed: {sorted(self.ledger.models)})")
        if not self.ledger.open():
            self.ledger.add(self.path, model, 0, 0, "refused-cap", 0.0)
            return self.refuse(429, f"the dollar cap of this cell (${self.ledger.cap_usd}) is used up")
        streamed = bool(payload.get("stream"))
        if streamed and self.path.startswith("/v1/chat/completions"):
            payload["stream_options"] = dict(payload.get("stream_options") or {}, include_usage=True)
        try:
            response = self.upstream("POST", json.dumps(payload).encode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read()
            self.ledger.add(self.path, model, 0, 0, f"upstream-{error.code}", time.perf_counter() - started)
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        with response:
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            if streamed:
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                usage = None
                for line in response:
                    usage = stream_usage(line) or usage
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                body = response.read()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                usage = json.loads(body).get("usage")
        input_tokens, output_tokens = usage_tokens(usage)
        self.ledger.add(self.path, model, input_tokens, output_tokens, "ok" if usage else "ok-no-usage", time.perf_counter() - started)

    # keeps the console quiet; the usage log has every call
    def log_message(self, *arguments):
        pass

# command line: python llmproxy.py --port 8771 --log usage.jsonl --cap 0.25...
# (the real key comes from the environment variable named by --key-variable)
def main():
    parser = argparse.ArgumentParser(description="Loopback proxy to the OpenAI API that counts tokens and enforces a model and a cap.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--cap", type=float, required=True, help="dollar cap of this cell")
    parser.add_argument("--models", nargs="+", default=["gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"])
    parser.add_argument("--price-input", type=float, default=0.4, help="dollars per million input tokens")
    parser.add_argument("--price-output", type=float, default=1.6, help="dollars per million output tokens")
    parser.add_argument("--key-variable", default="BENCH_OPENAI_KEY")
    arguments = parser.parse_args()
    ProxyHandler.api_key = os.environ.get(arguments.key_variable, "")
    if not ProxyHandler.api_key:
        print(f"the environment variable {arguments.key_variable} is empty", file=sys.stderr)
        return 2
    ProxyHandler.ledger = Ledger(arguments.models, arguments.price_input, arguments.price_output, arguments.cap, arguments.log)
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), ProxyHandler)
    server.serve_forever()
    return 0

if __name__ == "__main__":
    sys.exit(main())
