import os
import sys
import argparse
import subprocess
from paths import envs_dir

PYTHON = "cpython-3.12-windows-x86_64-none" if os.name == "nt" else "3.12"

# the python executable of a baseline environment (Windows or Linux layout)
def env_python(name):
    return envs_dir / name / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# creates benchmark/envs/<name>/.venv with uv and installs <name>.txt into it
def install(name):
    venv = envs_dir / name / ".venv"
    subprocess.run(["uv", "venv", str(venv), "--python", PYTHON, "--allow-existing"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(env_python(name)), "-r", str(envs_dir / f"{name}.txt")], check=True)
    subprocess.run([str(env_python(name)), "-c", "import sysconfig; print(sysconfig.get_platform())"], check=True)

# command line: python -m benchmark.envs.install flaml autogluon
def main():
    parser = argparse.ArgumentParser(description="Install the separate baseline environments.")
    parser.add_argument("names", nargs="+")
    for name in parser.parse_args().names:
        install(name)
    return 0

if __name__ == "__main__":
    sys.exit(main())
