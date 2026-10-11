import json
from llms.codex import CodexModel
from tools.coderunner import CodeRunner
from paths import config_dir, prompts_dir, sandbox_dir

class TrainAgent:
    def __init__(self):
        self.coder = CodexModel()
        self.runner = CodeRunner(file_name="train.py")
        self.last_reasoning = None
        self.last_generated_code = None
        self.configuration_path = config_dir / "configuration.json"
        plan_path = sandbox_dir / "trainingplan.md"

        with open(plan_path, 'r', encoding="utf-8") as plan:
            self.plan = plan.read()
        with open(self.configuration_path, 'r', encoding="utf-8") as f:
            self.runtime_config = json.load(f)
        with open(prompts_dir / "trainreasoningprompt.md", "r", encoding="utf-8") as f:
            self.reasoning_prompt = f.read()

    def train(self, max_attempts: int = 10):
        output = self.act()
        for _ in range(max_attempts):
            reasoning = self.reason(output=output)
            if reasoning.condition == "pass":
                return
            output = self.fix_code()
        raise RuntimeError(f"Training still failing after {max_attempts} fix attempts. Last problem: {self.last_reasoning}")

    def act(self) -> str:
        train_file = self.runtime_config['trainFile']
        val_file = self.runtime_config.get('valFile')
        header = f"Train Data Path: {train_file}\n"
        if val_file:
            header += f"Validation Data Path: {val_file}\n"
        header += "\n"
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
        generated_reasoning = self.coder.reason(self.reasoning_prompt, output["output"])
        self.last_reasoning = generated_reasoning.prompt
        return generated_reasoning
