import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from schemas import ReasoningResult
from paths import prompts_dir, env_path

class CodexModel:
    def __init__(self, model: str = "gpt-4.1-mini"):
        load_dotenv(env_path)
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(prompts_dir / "codexcodeprompt.md", "r", encoding="utf-8") as f:
            self._code_prompt = f.read()
        with open(prompts_dir / "codexfixprompt.md", "r", encoding="utf-8") as f:
            self._fix_prompt = f.read()

    def code(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._code_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content.strip()

    def fix_code(self, code: str, problem: str) -> str:
        user_message = f"### Original Code\n{code}\n\n### Problem\n{problem}"
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._fix_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return completion.choices[0].message.content.strip()

    def reason(self, system_prompt: str, user_content: str) -> ReasoningResult:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=ReasoningResult,
        )
        return completion.choices[0].message.parsed

    def tune_parameters(self, current_code: str, changes: str) -> str:
        prompt = (
            f"{self._code_prompt}\n\n"
            "Additional constraints:\n"
            "1. Return the full updated Python code.\n"
            "2. Only change parameter values in the existing training pipeline.\n"
            "3. Do not change model architecture, imports, data flow, preprocessing steps, function definitions, or file structure.\n"
            "4. Keep all non-parameter code exactly as is.\n"
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": current_code},
                {"role": "user", "content": changes},
            ],
        )
        return completion.choices[0].message.content

    def suggest_parameters(self, system_prompt: str, user_content: str) -> dict:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)