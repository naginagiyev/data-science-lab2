import os
import time
import logging
import threading
from benchmark import spend
from dotenv import dotenv_values
from smolagents import OpenAIModel
from tools.runfolder import now_text
from paths import env_file, spend_file

ESTIMATED_OUTPUT_TOKENS = 500
logger = logging.getLogger("automl.llm")

# raised when the agent may not make another call (run budget, global spend...
# cap or the stage deadline); the edatool session then stops and the...
# deterministic audit result is kept for its open items
class AgentBudgetError(RuntimeError):
    pass

# a smolagents OpenAI model for edatool's review agent: every call is checked...
# against the run budget, the global spend caps and a deadline, logged to the...
# run's llm.jsonl and written to the spend ledger; it is safe for the parallel...
# sessions of edatool (one lock)
class AgentModel(OpenAIModel):
    def __init__(self, settings, folder=None, stage="audit", max_cost_usd=None, ledger_path=None, deadline=None, model_id=None):
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(env_file).get("OPENAI_API_KEY") or "missing-key"
        options = {"temperature": settings.llm.temperature, "seed": settings.llm.seed}
        client = {"timeout": settings.llm.timeout_seconds, "max_retries": settings.llm.api_retries}
        super().__init__(model_id=model_id or settings.llm.model, api_key=key, client_kwargs=client, **options)
        self.settings = settings
        self.folder = folder
        self.stage = stage
        self.max_cost_usd = settings.llm.max_cost_per_run_usd if max_cost_usd is None else max_cost_usd
        self.ledger_path = ledger_path or spend_file
        self.deadline = deadline
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def price(self):
        prices = self.settings.llm.prices_per_million_tokens
        return prices.get(self.model_id) or max(prices.values(), key=lambda price: price["output"])

    def cost(self, input_tokens, output_tokens):
        price = self.price()
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    def check(self, messages):
        characters = sum(len(str(message)) for message in messages)
        estimate = self.cost(characters / 3, ESTIMATED_OUTPUT_TOKENS)
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise AgentBudgetError("the audit time budget is used up")
        if self.usage["cost_usd"] + estimate > self.max_cost_usd:
            raise AgentBudgetError(f"the agent budget of ${self.max_cost_usd} for this run is used up")
        allowed, reason = spend.check_allowed(estimate, self.settings.spend, self.ledger_path)
        if not allowed:
            raise AgentBudgetError(reason)

    def log(self, record):
        if self.folder is not None:
            self.folder.append("llm.jsonl", record)

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        with self.lock:
            self.check(messages)
        start = time.perf_counter()
        record = {"time": now_text(), "stage": self.stage, "prompt": "edatool-agent", "model": self.model_id, "cached": False}
        try:
            message = super().generate(messages, stop_sequences=stop_sequences, response_format=response_format, tools_to_call_from=tools_to_call_from, **kwargs)
        except Exception as error:
            with self.lock:
                self.usage["failed_calls"] += 1
                self.log(dict(record, ok=False, error=f"{type(error).__name__}: {error}", seconds=round(time.perf_counter() - start, 3), input_tokens=0, output_tokens=0, cost_usd=0.0))
            raise
        usage = getattr(message, "token_usage", None)
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        cost = self.cost(input_tokens, output_tokens)
        with self.lock:
            spend.record(cost, self.settings.spend.category, self.model_id, input_tokens, output_tokens, f"{self.stage}/edatool-agent", self.settings.spend, self.ledger_path)
            self.usage["calls"] += 1
            self.usage["input_tokens"] += input_tokens
            self.usage["output_tokens"] += output_tokens
            self.usage["cost_usd"] += cost
            self.log(dict(record, ok=True, error=None, seconds=round(time.perf_counter() - start, 3), input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost))
        return message
