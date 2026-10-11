import time
from edabench.legacy.eda import EDA
from paths import edabench_legacy_runs_dir
from edabench.datasets import local_dataset_path, local_datasets

# runs the old eda.py on one dataset and writes its report into runs/legacy
def run_legacy(dataset_path, target_col, task_type):
    edabench_legacy_runs_dir.mkdir(parents=True, exist_ok=True)
    eda = EDA(input_path=str(dataset_path), target_col=target_col, task_type=task_type)
    eda.output_path = str(edabench_legacy_runs_dir / f"{dataset_path.stem}eda.md")
    eda.run()
    return eda.output_path

# runs the old eda.py on the 4 local datasets and prints the time of each
def main():
    for dataset in local_datasets:
        start = time.perf_counter()
        output_path = run_legacy(local_dataset_path(dataset["file"]), dataset["target"], dataset["task"])
        print(f"{dataset['file']}: {output_path} ({time.perf_counter() - start:.1f} s)")

main()
