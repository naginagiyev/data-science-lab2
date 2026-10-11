import json
from pathlib import Path
from llms.generation import GenerationModel
from paths import prompts_dir, config_dir, sandbox_dir

class Planner:
    def __init__(self):
        self.configuration_path = config_dir / "configuration.json"
        self.processing_model = GenerationModel(prompts_dir / "processingplanprompt.md")
        self.training_model = GenerationModel(prompts_dir / "trainplanprompt.md")

    def _load_consultation(self):
        with open(self.configuration_path, "r", encoding="utf-8") as f:
            self.consultation = json.load(f)

    def _load_model_options(self):
        model_options_path = config_dir / "modeloptions.md"
        with open(model_options_path, "r", encoding="utf-8") as f:
            self.model_options = f.read()

    def _load_eda_report(self):
        stem = Path(self.consultation["dataFile"]).stem
        eda_path = sandbox_dir / f"{stem}eda.md"
        with open(eda_path, "r", encoding="utf-8") as f:
            self.eda_report = f.read()

    def _refresh_context(self):
        self._load_consultation()
        self._load_model_options()
        self._load_eda_report()
        sandbox_dir.mkdir(parents=True, exist_ok=True)

    def create_pre_processing_plan(self):
        self._refresh_context()
        query = (
            f"Dataset Path: {self.consultation['dataFile']}\n"
            f"Task Type: {self.consultation['taskType']}\n"
            f"Target Column: {self.consultation['targetCol']}\n\n"
            f"EDA Report:\n{self.eda_report}\n\n"
        )

        processing_plan = self.processing_model.generate(query=query)
        
        preprocessing_plan_path = sandbox_dir / "processingplan.md"
        with open(preprocessing_plan_path, "w", encoding="utf-8") as f:
            f.write(processing_plan)

        return preprocessing_plan_path

    def create_training_plan(self):
        self._refresh_context()
        query = (
            f"Train Data Path: {self.consultation['trainFile']}\n"
            f"Validation Data Path: {self.consultation.get('valFile', 'No Validation Data')}\n"
            f"Task Type: {self.consultation['taskType']}\n"
            f"Target Column: {self.consultation['targetCol']}\n\n"
            f"Explainable Model: {self.consultation.get('explainableModel')}\n"
            f"Model Options:\n{self.model_options}\n\n"
        )

        train_plan = self.training_model.generate(query=query)
        
        train_plan_path = sandbox_dir / "trainingplan.md"
        with open(train_plan_path, "w", encoding="utf-8") as f:
            f.write(train_plan)

        return train_plan_path
