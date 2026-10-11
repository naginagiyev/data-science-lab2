import json
import pandas as pd
from paths import prompts_dir
from schemas import TargetDetectionResult
from llms.generation import GenerationModel

class TargetDetector:
    def __init__(self):
        self.model = GenerationModel(
            prompts_dir / "targetdetector.md",
            response_format=TargetDetectionResult
        )

    def detect(self, dataset_path: str) -> str:
        df = pd.read_csv(dataset_path)
        col_samples = {col: df[col].dropna().unique()[:2].tolist() for col in df.columns}
        query = json.dumps({"columns": col_samples}, default=str)
        result = self.model.generate(query=query)
        return result.targetCol
