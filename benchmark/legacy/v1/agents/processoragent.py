import os
import json
import pandas as pd
from llms.codex import CodexModel
from tools.coderunner import CodeRunner
from paths import sandbox_dir, config_dir, data_dir, prompts_dir

class ProcessorAgent:
    def __init__(self):
        self.coder = CodexModel()
        self.runner = CodeRunner(file_name="processing.py")
        self.last_reasoning = None
        self.last_generated_code = None
        self.configuration_path = config_dir / "configuration.json"
        plan_path = sandbox_dir / "processingplan.md"

        with open(plan_path, 'r', encoding="utf-8") as plan:
            self.plan = plan.read()
        with open(self.configuration_path, 'r', encoding="utf-8") as configuration_file:
            self.configuration = json.load(configuration_file)
        with open(prompts_dir / "processorreasoningprompt.md", "r", encoding="utf-8") as f:
            self.reasoning_prompt = f.read()

    def preprocess(self, max_attempts: int = 10):
        output = self.act()
        for _ in range(max_attempts):
            reasoning = self.reason(output=output)
            if reasoning.condition == "pass":
                return
            output = self.fix_code()
        raise RuntimeError(f"Preprocessing still failing after {max_attempts} fix attempts. Last problem: {self.last_reasoning}")

    def act(self) -> str:
        header = f"Dataset Path: {self.configuration['dataFile']}\n\n"
        generated_code = self.coder.code(header + self.plan)
        self.last_generated_code = generated_code
        self.runner.add(generated_code)
        return self.runner.get_output()

    def fix_code(self) -> str:
        fixed_code = self.coder.fix_code(self.last_generated_code, self.last_reasoning)
        self.last_generated_code = fixed_code
        self.runner.add(fixed_code)
        return self.runner.get_output()

    def reason(self, output: str) -> dict:
        if output["operation"] == "success":
            processed_columns = self.get_data_info()
            success_payload = {
                "taskType": self.configuration.get("taskType"),
                "targetCol": self.configuration.get("targetCol"),
                "processedColumns": processed_columns,
            }
            generated_reasoning = self.coder.reason(
                self.reasoning_prompt, json.dumps(success_payload, indent=2, default=str)
            )
        else:
            generated_reasoning = self.coder.reason(self.reasoning_prompt, output["output"])
        self.last_reasoning = generated_reasoning.prompt
        return generated_reasoning

    def get_data_info(self) -> dict:
        data_file = self.configuration['dataFile']
        base, ext = os.path.splitext(os.path.basename(data_file))
        processed_basename = f"{base}_processed{ext}"
        processed_data_path = data_dir / processed_basename
        processed_data = pd.read_csv(processed_data_path)

        self.configuration['processedDataFile'] = processed_basename
        with open(self.configuration_path, 'w', encoding="utf-8") as f:
            json.dump(self.configuration, f, indent=2)

        return {"rowCount": len(processed_data), "columns": {
            col: {
                "non_null_count": int(processed_data[col].notnull().sum()),
                "dtype": str(processed_data[col].dtype)
            }
            for col in processed_data.columns
        }}