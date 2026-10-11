import sys

# do not let the pycaches to appear
sys.dont_write_bytecode = True

import time
import json
import logging
import traceback
from pathlib import Path
from agents.planner import Planner
from agents.trainagent import TrainAgent
from agents.configfiller import ConfigFiller
from agents.processoragent import ProcessorAgent
from agents.targetdetector import TargetDetector
from tools.datasplitter import split as split_data
from agents.evaluationagent import EvaluationAgent
from main import run_eda, infer_task_type_from_target
from paths import config_dir, sandbox_dir, models_dir

# plain debug runner for the pipeline: every step is logged to the terminal and to sandbox/debug.log

sandbox_dir.mkdir(parents=True, exist_ok=True)
log_path = sandbox_dir / "debug.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
)

for noisy in ("httpx", "httpcore", "openai", "urllib3", "matplotlib", "PIL", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("main2")
configuration_path = config_dir / "configuration.json"


def read_config() -> dict:
    if not configuration_path.exists():
        return {}
    with open(configuration_path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_config_diff(before: dict, after: dict):
    changed = False
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changed = True
            log.debug(f"  config[{key}]: {before.get(key)!r} -> {after.get(key)!r}")
    if not changed:
        log.debug("  config unchanged")


def list_dir(path: Path):
    if not path.exists():
        log.debug(f"  {path} does not exist")
        return
    for item in sorted(path.iterdir()):
        size = item.stat().st_size if item.is_file() else "-"
        log.debug(f"  {item.name} ({size} bytes)")


def step(name: str, fn, *args, **kwargs):
    log.info("=" * 70)
    log.info(f"STEP: {name}")
    if args or kwargs:
        log.debug(f"  args={args} kwargs={kwargs}")
    before = read_config()
    start = time.time()
    try:
        result = fn(*args, **kwargs)
    except Exception:
        log.error(f"STEP FAILED: {name} after {time.time() - start:.2f}s")
        log.error(traceback.format_exc())
        raise
    log.info(f"STEP DONE: {name} in {time.time() - start:.2f}s")
    log.debug(f"  result={result!r}")
    log_config_diff(before, read_config())
    return result


def run_shared_pipeline():
    planner = Planner()
    preprocessing_path = step("Planner.createPreProcessingPlan", planner.create_pre_processing_plan)
    log.info(f"Processing plan saved to {preprocessing_path}")
    step("ProcessorAgent.preprocess", ProcessorAgent().preprocess)
    step("split_data", split_data)
    training_path = step("Planner.createTrainingPlan", planner.create_training_plan)
    log.info(f"Training plan saved to {training_path}")
    step("TrainAgent.train", TrainAgent().train)
    step("EvaluationAgent.tune", EvaluationAgent().tune)


def main():
    start = time.time()
    log.info(f"Debug log file: {log_path}")
    log.debug(f"Initial configuration: {json.dumps(read_config(), indent=2)}")

    while True:
        dataset_path = input("Path to your dataset: ").strip().strip('"')
        if not dataset_path:
            log.warning("No path given, exiting")
            return
        if not Path(dataset_path).exists():
            log.warning(f"Dataset not found at {dataset_path}")
            continue
        break
    log.info(f"Dataset: {dataset_path}")

    detected_target = step("TargetDetector.detect", TargetDetector().detect, dataset_path)
    task_type = step("infer_task_type_from_target", infer_task_type_from_target, dataset_path, detected_target)
    step("EDA", run_eda, dataset_path, detected_target, task_type)
    step("ConfigFiller.fill", ConfigFiller().fill, dataset_path, detected_target, task_type)

    log.debug(f"Configuration after setup: {json.dumps(read_config(), indent=2)}")
    run_shared_pipeline()

    log.info("=" * 70)
    log.info(f"Final configuration: {json.dumps(read_config(), indent=2)}")
    log.info("Sandbox contents:")
    list_dir(sandbox_dir)
    log.info("Models contents:")
    list_dir(models_dir)
    log.info(f"All done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
