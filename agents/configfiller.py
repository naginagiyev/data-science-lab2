import json
from pathlib import Path
from schemas import ProjectConfig
from llms.generation import GenerationModel
from paths import prompts_dir, config_dir, sandbox_dir

class ConfigFiller:
    def __init__(self):
        self.model = GenerationModel(
            prompts_dir / "configfiller.md",
            response_format=ProjectConfig,
        )
        self.metrics_path = config_dir / "metrics.json"
        self.allowed_metrics = self.load_allowed_metrics()

    def load_allowed_metrics(self) -> list[str]:
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics_by_group = json.load(f)
        metrics = []
        for group_metrics in metrics_by_group.values():
            for metric in group_metrics:
                if metric not in metrics:
                    metrics.append(metric)
        return metrics

    def normalize_metric(self, metric: str | None) -> str | None:
        if metric is None:
            return None
        normalized_map = {item.lower(): item for item in self.allowed_metrics}
        return normalized_map.get(str(metric).strip().lower())

    def fill(self, dataset_path: str, target_col: str, task_type: str) -> dict:
        stem = Path(dataset_path).stem
        eda_path = sandbox_dir / f"{stem}eda.md"

        with open(eda_path, "r", encoding="utf-8") as f:
            eda_report = f.read()

        metrics_text = ", ".join(self.allowed_metrics)
        query = (
            f"Task Type: {task_type}\n"
            f"Allowed Metrics (use exact naming): {metrics_text}\n"
            "desiredMetric must be exactly one value from Allowed Metrics or null.\n\n"
            f"EDA Report:\n{eda_report}"
        )

        result = self.model.generate(query=query).model_dump()
        filled = {
            "dataFile": Path(dataset_path).name,
            "taskType": task_type,
            "targetCol": target_col,
            **result,
        }
        filled["desiredMetric"] = self.normalize_metric(filled.get("desiredMetric"))

        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "configuration.json", "w", encoding="utf-8") as f:
            json.dump(filled, f, indent=4)

        return filled
