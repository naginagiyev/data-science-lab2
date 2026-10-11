import json
import pandas as pd
import onnxruntime as ort
from llms.codex import CodexModel
from tools.datareader import load_data
from tools.metrics import compute_score
from tools.coderunner import CodeRunner
from paths import config_dir, models_dir, prompts_dir, sandbox_dir, data_dir

class EvaluationAgent:
    def __init__(self):
        self.coder = CodexModel()
        self.runner = CodeRunner(file_name="train.py")

        plan_path = prompts_dir / "evaluationprompt.md"
        with open(plan_path, 'r', encoding="utf-8") as plan:
            self.evaluation_prompt = plan.read()

        configuration_path = config_dir / "configuration.json"
        with open(configuration_path, 'r', encoding="utf-8") as f:
            self.configuration = json.load(f)
        self.consultation = self.configuration

        self.metric = self.consultation.get('desiredMetric')
        self.target = self.consultation.get('targetCol')
        self.min_metric = self.consultation.get('minScoreRequirement')

        self.runtime_config = self.configuration

        self.train_data_path = data_dir / self.runtime_config.get('trainFile')
        self.test_data_path = data_dir / self.runtime_config.get('testFile')

        constants_path = config_dir / "constants.json"
        with open(constants_path, 'r', encoding="utf-8") as f:
            self.constants = json.load(f)
        
        self.tuning_iterations = self.constants.get('tuningIterations')
        self.param_history = []
        self.previous_scores = None
        self.lower_is_better = str(self.metric).lower() in {"mae", "mse", "rmse", "mape", "log-loss"}

    def load_model(self):
        self.session = ort.InferenceSession(models_dir / "model.onnx")     

    def get_predictions(self, input: pd.DataFrame) -> pd.DataFrame:
        feature_df = input.drop(columns=[self.target])
        model_inputs = self.session.get_inputs()
        if len(model_inputs) == 1:
            input_feed = {model_inputs[0].name: feature_df.values.astype("float32")}
        else:
            input_feed = {inp.name: feature_df[[inp.name]].values.astype("float32") for inp in model_inputs}
        model_outputs = self.session.run(None, input_feed)
        return pd.DataFrame(model_outputs[0], columns=["prediction"])

    def evaluate(self, df: pd.DataFrame) -> float:
        predictions = self.get_predictions(df)
        true_values = df[self.target]
        train_score = compute_score(
            metric=str(self.metric).lower(),
            true=true_values, pred=predictions["prediction"])
        return round(float(train_score), 2)
    
    def read_train_code(self) -> str:
        with open(sandbox_dir / "train.py", "r") as code:
            code = code.read()

        marker = "# Generated code starts here"
        return code.split(marker, 1)[1]

    def act(self, param_changes: dict) -> None:
        train_code = self.read_train_code()    

        tuned_code = self.coder.tune_parameters(
            current_code=f"Training code:\n{train_code}", 
            changes=f"Changes to make:\n{json.dumps(param_changes, ensure_ascii=False)}"
            )
        
        self.runner.add(tuned_code)
        return self.runner.get_output()   

    def tune(self):
        train_data = load_data(self.train_data_path, self.target)
        test_data = load_data(self.test_data_path, self.target)

        for iter in range(self.tuning_iterations):
            self.load_model()
            train_score = self.evaluate(train_data)
            test_score = self.evaluate(test_data)


            met_min = (train_score <= self.min_metric and test_score <= self.min_metric) if self.lower_is_better else (train_score >= self.min_metric and test_score >= self.min_metric)
            if met_min:
                self.param_history.append({"finalScores": {"train": train_score, "test": test_score}})
                break

            if self.param_history and self.previous_scores is not None:
                self.param_history[-1]["scoreBefore"] = self.previous_scores
                self.param_history[-1]["scoreAfter"] = {"train": train_score, "test": test_score}
                self.param_history[-1]["helped"] = (
                    train_score <= self.previous_scores["train"] and
                    test_score <= self.previous_scores["test"]
                ) if self.lower_is_better else (
                    train_score >= self.previous_scores["train"] and
                    test_score >= self.previous_scores["test"]
                )

            if iter == 0:
                query = (
                    f"Train Code:\n{self.read_train_code()}\n\n"
                    f"{self.metric.upper()} Train Score: {train_score}\n"
                    f"{self.metric.upper()} Test Score: {test_score}\n"
                    f"Min {self.metric.upper()} Score To Hit: {self.min_metric}\n\n"
                )
            else:
                query = (
                    f"Parameter Change History:\n{json.dumps(self.param_history, ensure_ascii=False)}\n\n"
                    f"{self.metric.upper()} Train Score: {train_score}\n"
                    f"{self.metric.upper()} Test Score: {test_score}\n"
                    f"Min {self.metric.upper()} Score To Hit: {self.min_metric}\n\n"
                )

            tuning_result = self.coder.suggest_parameters(
                system_prompt=self.evaluation_prompt,
                user_content=query
            )
            param_changes = tuning_result
            self.param_history.append({"iteration": iter + 1, "parameterChanges": param_changes})
            self.previous_scores = {"train": train_score, "test": test_score}
            self.act(param_changes)

        with open(sandbox_dir / "tuning.json", "w", encoding="utf-8") as f:
            json.dump(self.param_history, f, indent=2, ensure_ascii=False)