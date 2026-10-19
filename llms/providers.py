import os
import json
import openai
from paths import env_file
from pydantic import BaseModel
from dotenv import dotenv_values
from openai.lib._parsing._completions import type_to_response_format_param

# the answer of one provider call: text, token counts and the model that...
# answered
class Completion(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    model: str

# raised when the provider rejects a request parameter (for example...
# temperature on some newer models), so the client can retry without it
class UnsupportedParameter(Exception):
    def __init__(self, name, message):
        super().__init__(message)
        self.name = name

# calls OpenAI chat completions; with a schema it uses strict structured...
# outputs and returns the raw JSON text, so the client validates it; the key is...
# read from the environment or .env without being copied into os.environ
class OpenAIProvider:
    def __init__(self, timeout_seconds=60):
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(env_file).get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=key, timeout=timeout_seconds, max_retries=0)
        self.transient = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)

    def complete(self, messages, schema, model, options):
        try:
            if schema is not None:
                options = dict(options, response_format=type_to_response_format_param(schema))
            response = self.client.chat.completions.create(model=model, messages=messages, **options)
        except openai.BadRequestError as error:
            text = str(error).lower()
            for name in options:
                if name in text and ("unsupported" in text or "not support" in text):
                    raise UnsupportedParameter(name, str(error)) from error
            raise
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
        )

# a provider for tests: answers come from a list (in order) or from a function...
# of (prompt text, call number); an exception in the list is raised instead
class FakeProvider:
    def __init__(self, answers, model="fake-model"):
        self.answers = answers
        self.model = model
        self.calls = []
        self.transient = (ConnectionError, TimeoutError)

    def complete(self, messages, schema, model, options):
        self.calls.append({"messages": messages, "schema": schema, "model": model, "options": dict(options)})
        if callable(self.answers):
            answer = self.answers(messages[-1]["content"], len(self.calls))
        else:
            answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        prompt = " ".join(message["content"] for message in messages)
        return Completion(text=text, input_tokens=max(1, len(prompt) // 4), output_tokens=max(1, len(text) // 4), model=model)
