import re
import json
import time
import hashlib
import logging
from benchmark import spend
from pydantic import ValidationError
from tools.runfolder import now_text
from paths import llm_cache_dir, prompts_dir, spend_file
from llms.providers import Completion, OpenAIProvider, UnsupportedParameter
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

ESTIMATED_OUTPUT_TOKENS = 1500
PLACEHOLDER = re.compile(r"\{(\w+)\}")
logger = logging.getLogger("automl.llm")

# fills the {name} placeholders of a prompt with the inputs; braces that are...
# not a known placeholder (for example in JSON examples) stay as they are
def render(template, inputs):
    return PLACEHOLDER.sub(lambda match: str(inputs[match.group(1)]) if match.group(1) in inputs else match.group(0), template)

# returns a short sha256 hash of a text
def text_hash(text, length=16):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

# turns a pydantic validation error into a few short lines for the LLM
def short_errors(error):
    lines = []
    for item in error.errors()[:10]:
        place = ".".join(str(part) for part in item["loc"]) or "answer"
        lines.append(f"- {place}: {item['msg']}")
    return "\n".join(lines)

# the only way to call an LLM: renders a prompt, uses the disk cache, enforces...
# the run and global budgets, retries API and schema errors, validates the...
# answer and logs every call; ask() returns the validated object or None
class LlmClient:
    def __init__(self, settings, folder=None, provider=None, ledger_path=None, cache_dir=None, prompt_dir=None, retry_wait=1.0):
        self.settings = settings
        self.llm = settings.llm
        self.folder = folder
        self.provider = provider
        self.ledger_path = ledger_path or spend_file
        self.cache_dir = cache_dir or llm_cache_dir
        self.prompt_dir = prompt_dir or prompts_dir
        self.retry_wait = retry_wait
        self.dropped_options = set()
        self.usage = {"calls": 0, "cached_calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def get_provider(self):
        if self.provider is None:
            self.provider = OpenAIProvider(self.llm.timeout_seconds)
        return self.provider

    def price(self, model):
        prices = self.llm.prices_per_million_tokens
        if model in prices:
            return prices[model]
        logger.warning(f"No price for model {model}; using the highest known price.")
        return max(prices.values(), key=lambda price: price["output"])

    def cost(self, model, input_tokens, output_tokens):
        price = self.price(model)
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    def allowed(self, messages, model):
        characters = sum(len(message["content"]) for message in messages)
        estimate = self.cost(model, characters / 3, ESTIMATED_OUTPUT_TOKENS)
        if self.usage["calls"] >= self.llm.max_calls_per_run:
            return False, f"the run reached max_calls_per_run ({self.llm.max_calls_per_run})"
        if self.usage["cost_usd"] + estimate > self.llm.max_cost_per_run_usd:
            return False, f"the run would cross max_cost_per_run_usd (${self.llm.max_cost_per_run_usd})"
        return spend.check_allowed(estimate, self.settings.spend, self.ledger_path)

    def call_provider(self, messages, schema, model, options):
        provider = self.get_provider()
        wait = wait_exponential(multiplier=self.retry_wait, max=30)
        retrying = Retrying(stop=stop_after_attempt(self.llm.api_retries + 1), wait=wait, retry=retry_if_exception_type(provider.transient), reraise=True)
        for attempt in retrying:
            with attempt:
                try:
                    return provider.complete(messages, schema, model, options)
                except UnsupportedParameter as error:
                    logger.warning(f"The provider rejected '{error.name}'; retrying without it.")
                    self.dropped_options.add(error.name)
                    options = {name: value for name, value in options.items() if name != error.name}
                    return provider.complete(messages, schema, model, options)

    def record_usage(self, record, context=None):
        if record["cached"]:
            self.usage["cached_calls"] += 1
        elif record["input_tokens"] or record["output_tokens"]:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += record["input_tokens"]
            self.usage["output_tokens"] += record["output_tokens"]
            self.usage["cost_usd"] += record["cost_usd"]
        if not record["ok"]:
            self.usage["failed_calls"] += 1
        if self.folder is not None:
            self.folder.append("llm.jsonl", record)
        if context is not None:
            context.llm = dict(self.usage)

    def complete(self, messages, schema, stage, prompt_name, prompt_hash):
        model = self.llm.model
        options = {"temperature": self.llm.temperature, "seed": self.llm.seed}
        options = {name: value for name, value in options.items() if name not in self.dropped_options}
        schema_json = schema.model_json_schema() if schema is not None else None
        key = text_hash(json.dumps({"model": model, "options": options, "schema": schema_json, "messages": messages}, sort_keys=True), 64)
        record = {
            "time": now_text(), "stage": stage, "prompt": prompt_name, "prompt_hash": prompt_hash, "input_hash": key[:16],
            "model": model, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "seconds": 0.0, "cached": False,
            "ok": False, "error": None, "messages": messages, "answer": None,
        }
        cache_file = self.cache_dir / key[:2] / f"{key}.json"
        if self.llm.cache and cache_file.exists():
            completion = Completion.model_validate_json(cache_file.read_text(encoding="utf-8"))
            record.update(cached=True, input_tokens=completion.input_tokens, output_tokens=completion.output_tokens, answer=completion.text)
            return completion, record
        allowed, reason = self.allowed(messages, model)
        if not allowed:
            record["error"] = f"budget: {reason}"
            logger.warning(f"LLM call refused ({stage}/{prompt_name}): {reason}")
            return None, record
        start = time.perf_counter()
        try:
            completion = self.call_provider(messages, schema, model, options)
        except Exception as error:
            record.update(error=f"{type(error).__name__}: {error}", seconds=round(time.perf_counter() - start, 3))
            logger.warning(f"LLM call failed ({stage}/{prompt_name}): {record['error']}")
            return None, record
        cost = self.cost(model, completion.input_tokens, completion.output_tokens)
        spend.record(cost, self.settings.spend.category, model, completion.input_tokens, completion.output_tokens, f"{stage}/{prompt_name}", self.settings.spend, self.ledger_path)
        if self.llm.cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(completion.model_dump_json(), encoding="utf-8")
        record.update(
            input_tokens=completion.input_tokens, output_tokens=completion.output_tokens, cost_usd=cost,
            seconds=round(time.perf_counter() - start, 3), answer=completion.text,
        )
        return completion, record

    def validate(self, text, schema, check):
        if schema is None:
            result = text
        else:
            try:
                result = schema.model_validate_json(text)
            except ValidationError as error:
                return None, short_errors(error)
        problems = check(result) if check is not None else []
        if problems:
            return None, "\n".join(f"- {problem}" for problem in problems)
        return result, None

    def ask(self, prompt_name, inputs, schema, stage, context=None, check=None):
        template = (self.prompt_dir / f"{prompt_name}.md").read_text(encoding="utf-8")
        messages = [{"role": "user", "content": render(template, inputs)}]
        for attempt in range(self.llm.schema_retries + 1):
            completion, record = self.complete(messages, schema, stage, prompt_name, text_hash(template))
            record["attempt"] = attempt
            if completion is None:
                self.record_usage(record, context)
                return None
            result, errors = self.validate(completion.text, schema, check)
            record["ok"] = errors is None
            record["error"] = None if errors is None else f"invalid answer:\n{errors}"
            self.record_usage(record, context)
            if errors is None:
                return result
            logger.info(f"Invalid LLM answer ({stage}/{prompt_name}, attempt {attempt + 1}):\n{errors}")
            messages = messages + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"Your answer was not valid:\n{errors}\nReturn the full corrected answer."},
            ]
        return None
