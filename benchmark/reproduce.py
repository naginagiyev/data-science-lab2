import sys
import argparse
import subprocess
from paths import root_dir
from benchmark.datasets import GROUP_A, GROUP_C, GROUP_D

TYPES = ["ids", "packed", "dates", "booleans", "spellings", "mixed_units", "units", "impossible", "fake_missing", "plain_missing", "layout"]
PILOT_GRIDS = ["pilot-main", "pilot-leaks", "pilot-ablations", "pilot-types", "pilot-seed", "pilot-natural", "pilot-budget", "reading"]
FULL_GRIDS = ["main", "leaks", "ablations", "types", "budgets"]

# runs one python module of this project and stops on an error
def module(name, *arguments):
    print(f"> python -m {name} {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, "-m", name, *arguments], cwd=root_dir, check=True)

# writes the benchmark files of Groups A, C and D and the single-type files...
# (existing cells are kept)
def benchmark_files():
    module("benchmark.materialize", "--datasets", *GROUP_A, "--levels", "0", "1", "2", "3", "--seeds", "0", "1", "2")
    for mess in TYPES:
        module("benchmark.materialize", "--datasets", *GROUP_A, "--levels", "3", "--seeds", "0", "--mess", mess)
    module("benchmark.materialize", "--datasets", *GROUP_C, "--levels", "0", "--seeds", "0", "1", "2")
    module("benchmark.materialize", "--datasets", *GROUP_D, "--levels", "0", "--seeds", "0", "1", "2", "--group", "D")
    module("benchmark.materialize", "--manifest")

# reproduces one grid from scratch: smoke (dev samples), pilot (every pilot...
# grid, the repair checks and the analysis) or one full grid; finished cells...
# are skipped, so a stopped run can be started again
def reproduce(grid):
    if grid == "smoke":
        module("benchmark.materialize", "--datasets", "ibmchurn", "houses", "--levels", "0", "3", "--seeds", "0", "--group", "dev5k", "--sample-rows", "6250")
        module("benchmark.runner", "--grid", "smoke")
        return
    benchmark_files()
    if grid == "pilot":
        for name in PILOT_GRIDS:
            module("benchmark.runner", "--grid", name)
        module("benchmark.repairs")
        module("benchmark.repairs", "--from-grid", "pilot-types")
        module("benchmark.framereval")
        module("benchmark.projection", "--predict")
        module("benchmark.runner", "--grid", "pilot-main", "--holdout")
        module("benchmark.figures")
        module("benchmark.resultsdoc")
        module("benchmark.projection")
        return
    module("benchmark.runner", "--grid", grid)

# command line: python -m benchmark.reproduce --grid smoke|pilot|main|...
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Reproduce a benchmark grid with one command.")
    parser.add_argument("--grid", required=True, choices=["smoke", "pilot", *FULL_GRIDS])
    reproduce(parser.parse_args().grid)
    return 0

if __name__ == "__main__":
    sys.exit(main())
