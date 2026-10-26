from pathlib import Path

# every path of the project, built from the project root so the code works...
# from any working folder
root_dir = Path(__file__).resolve().parent
env_file = root_dir / ".env"

configuration_dir = root_dir / "configuration"
settings_file = configuration_dir / "settings.json"
metrics_file = configuration_dir / "metrics.json"
tokens_file = configuration_dir / "tokens.json"
audit_file = configuration_dir / "audit.json"

prompts_dir = root_dir / "prompts"
data_dir = root_dir / "data"
runs_dir = root_dir / "runs"
docs_dir = root_dir / "docs"
progress_file = docs_dir / "progress.md"

cache_dir = root_dir / "cache"
llm_cache_dir = cache_dir / "llm"
openml_cache_dir = cache_dir / "openml"
natural_cache_dir = cache_dir / "natural"
benchmark_files_dir = cache_dir / "benchmark" / "files"
benchmark_splits_dir = cache_dir / "benchmark" / "splits"
benchmark_outputs_dir = cache_dir / "benchmark" / "outputs"
workdirs_dir = cache_dir / "w"
backend_models_dir = cache_dir / "b"
logs_dir = cache_dir / "logs"

benchmark_dir = root_dir / "benchmark"
systems_dir = benchmark_dir / "systems"
systems_registry_file = systems_dir / "registry.json"
envs_dir = benchmark_dir / "envs"
grids_dir = benchmark_dir / "grids"
results_dir = benchmark_dir / "results"
figures_dir = results_dir / "figures"
tables_file = results_dir / "tables.md"
spend_file = results_dir / "spend.json"
