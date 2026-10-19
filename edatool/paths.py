from pathlib import Path

# paths of the package files, built from this file so they also work after...
# installation
package_dir = Path(__file__).resolve().parent
config_dir = package_dir / "config"
prompts_dir = package_dir / "agent" / "prompts"
thresholds_file = config_dir / "thresholds.json"
tokens_file = config_dir / "tokens.json"
catalog_file = config_dir / "catalog.json"
agent_file = config_dir / "agent.json"
