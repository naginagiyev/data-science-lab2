from pathlib import Path

# every path of the project, built from the project root so the code works...
# from any working folder
root_dir = Path(__file__).resolve().parent
env_file = root_dir / ".env"
progress_file = root_dir / "progress.md"

configuration_dir = root_dir / "configuration"
settings_file = configuration_dir / "settings.json"
metrics_file = configuration_dir / "metrics.json"

prompts_dir = root_dir / "prompts"
data_dir = root_dir / "data"
runs_dir = root_dir / "runs"
docs_dir = root_dir / "docs"
decisions_file = docs_dir / "decisions.md"
deviations_file = docs_dir / "deviations.md"

cache_dir = root_dir / "cache"
llm_cache_dir = cache_dir / "llm"
openml_cache_dir = cache_dir / "openml"
edatool_cache_dir = cache_dir / "edatool"
edatool_runs_dir = cache_dir / "edatool-runs"
benchmark_files_dir = cache_dir / "benchmark" / "files"
benchmark_splits_dir = cache_dir / "benchmark" / "splits"
benchmark_outputs_dir = cache_dir / "benchmark" / "outputs"
workdirs_dir = cache_dir / "w"
backend_models_dir = cache_dir / "b"
performance_dir = cache_dir / "performance"
logs_dir = cache_dir / "logs"

benchmark_dir = root_dir / "benchmark"
systems_dir = benchmark_dir / "systems"
systems_registry_file = systems_dir / "registry.json"
envs_dir = benchmark_dir / "envs"
grids_dir = benchmark_dir / "grids"
legacy_dir = benchmark_dir / "legacy"
v1_dir = legacy_dir / "v1"
results_dir = benchmark_dir / "results"
figures_dir = results_dir / "figures"
tables_dir = results_dir / "tables"
spend_file = results_dir / "spend.json"

edabench_dir = root_dir / "edabench"
edabench_labels_dir = edabench_dir / "labels"
edabench_legacy_dir = edabench_dir / "legacy"
edabench_results_dir = edabench_dir / "results"
edabench_runs_dir = cache_dir / "edabench"
edabench_legacy_runs_dir = edabench_runs_dir / "legacy"

tests_dir = root_dir / "tests"
fixtures_dir = tests_dir / "fixtures"
