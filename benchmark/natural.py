import io
import sys
import gzip
import json
import hashlib
import zipfile
import argparse
import datetime
import pandas as pd
import urllib.request
from paths import natural_cache_dir, results_dir
from benchmark.datasets import BINARY, REGRESSION, load_dataset

MANIFEST = results_dir / "natural.json"
USER_AGENT = "Mozilla/5.0 (research download; data-science-lab2)"
RAHA = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/"
UCI_BANK = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_COUPON = "https://archive.ics.uci.edu/static/public/603/in+vehicle+coupon+recommendation.zip"
UCI_STUDENT = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
UCI_DIABETES = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
EUROPE_PMC_HOTELS = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6297060/supplementaryFiles"
AIRBNB_AUSTIN = "https://data.insideairbnb.com/united-states/tx/austin/2026-09-17/data/listings.csv.gz"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"

# the naturally messy and leaky datasets (docs/natural-data.md): kind (mess,...
# leak or both), task, target, licence, source, citation and the files to fetch
NATURAL_DATASETS = {
    "beers": {
        "kind": "mess", "task": REGRESSION, "target": "abv",
        "licence": "ODbL 1.0 (database) and DbCL 1.0 (contents)", "licence_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "source": "https://github.com/BigDaMa/raha/tree/master/datasets/beers",
        "citation": "J.-N. Hould, Craft Beers Dataset (Kaggle, 2017); dirty and clean versions from Mahdavi et al., Raha, SIGMOD 2019",
        "files": {"dirty.csv": {"url": RAHA + "beers/dirty.csv"}, "clean.csv": {"url": RAHA + "beers/clean.csv"}},
    },
    "coupon": {
        "kind": "mess", "task": BINARY, "target": "Y", "licence": "CC BY 4.0", "licence_url": CC_BY,
        "source": "https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation",
        "citation": "T. Wang et al., A Bayesian framework for learning rule sets for interpretable classification, JMLR 18, 2017",
        "files": {"in-vehicle-coupon-recommendation.csv": {"url": UCI_COUPON, "members": ["in-vehicle-coupon-recommendation.csv"]}},
    },
    "hr-analytics": {
        "kind": "mess", "task": BINARY, "target": "LookingForJobChange", "licence": "CC0 (OpenML)", "licence_url": "https://www.openml.org/d/46935",
        "source": "https://www.openml.org/d/46935", "citation": "HR Analytics: Job Change of Data Scientists; OpenML 46935 (TabArena)",
        "files": {"hr-analytics.csv": {"openml": "HR_Analytics_Job_Change_of_Data_Scientists"}},
    },
    "bank-additional": {
        "kind": "leak", "task": BINARY, "target": "y", "licence": "CC BY 4.0", "licence_url": CC_BY,
        "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "citation": "S. Moro, P. Cortez, P. Rita, Decision Support Systems 62, 2014; UCI, doi:10.24432/C5K306",
        "files": {
            "bank-additional-full.csv": {"url": UCI_BANK, "members": ["bank-additional.zip", "bank-additional/bank-additional-full.csv"], "sep": ";"},
            "bank-additional-names.txt": {"url": UCI_BANK, "members": ["bank-additional.zip", "bank-additional/bank-additional-names.txt"]},
        },
    },
    "titanic": {
        "kind": "leak", "task": BINARY, "target": "survived", "licence": "Public (OpenML)", "licence_url": "https://www.openml.org/d/40945",
        "source": "https://www.openml.org/d/40945", "citation": "F. E. Harrell Jr., titanic3 (Vanderbilt Biostatistics); OpenML 40945",
        "files": {"titanic.csv": {"openml": "Titanic"}},
    },
    "student-performance": {
        "kind": "leak", "task": REGRESSION, "target": "G3", "licence": "CC BY 4.0", "licence_url": CC_BY,
        "source": "https://archive.ics.uci.edu/dataset/320/student+performance",
        "citation": "P. Cortez, A. Silva, Using data mining to predict secondary school student performance, FUBUTEC 2008; UCI, doi:10.24432/C5TG7T",
        "files": {
            "student-por.csv": {"url": UCI_STUDENT, "members": ["student.zip", "student-por.csv"], "sep": ";"},
            "student.txt": {"url": UCI_STUDENT, "members": ["student.zip", "student.txt"]},
        },
    },
    "diabetes130": {
        "kind": "both", "task": BINARY, "target": "readmitted", "licence": "CC BY 4.0", "licence_url": CC_BY,
        "source": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "citation": "B. Strack et al., BioMed Research International, 2014; UCI, doi:10.24432/C5230J",
        "files": {
            "diabetic_data.csv": {"url": UCI_DIABETES, "members": ["diabetic_data.csv"]},
            "IDS_mapping.csv": {"url": UCI_DIABETES, "members": ["IDS_mapping.csv"]},
        },
    },
    "hotels": {
        "kind": "both", "task": BINARY, "target": "IsCanceled", "licence": "CC BY 4.0", "licence_url": CC_BY,
        "source": "https://doi.org/10.1016/j.dib.2018.11.126",
        "citation": "N. Antonio, A. de Almeida, L. Nunes, Hotel booking demand datasets, Data in Brief 22, 2019",
        "files": {
            "H1.csv": {"url": EUROPE_PMC_HOTELS, "members": ["mmc2.zip", "H1.csv"]},
            "H2.csv": {"url": EUROPE_PMC_HOTELS, "members": ["mmc2.zip", "H2.csv"]},
        },
    },
    "airbnb-austin": {
        "kind": "both", "task": REGRESSION, "target": "price", "licence": "CC BY 4.0", "licence_url": CC_BY,
        "source": "https://insideairbnb.com/get-the-data/", "citation": "Inside Airbnb, Austin, snapshot 2026-09-17",
        "files": {"listings.csv": {"url": AIRBNB_AUSTIN, "gzip": True}},
    },
}

# downloads a URL once per run and keeps its bytes in the store
def fetch(url, store):
    if url not in store:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response:
            store[url] = response.read()
    return store[url]

# unpacks downloaded bytes: gzip first, then one member per zip level, so a...
# list of members walks into nested zip files
def unpack(data, members=(), compressed=False):
    data = gzip.decompress(data) if compressed else data
    for member in members:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            data = archive.read(member)
    return data

# the bytes of one dataset file: an OpenML table as raw text CSV, or the...
# unpacked download
def file_bytes(spec, store):
    if "openml" in spec:
        return load_dataset(spec["openml"]).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return unpack(fetch(spec["url"], store), spec.get("members", ()), spec.get("gzip", False))

# downloads the files of one dataset into cache/natural/<name>/ (kept if they...
# exist) and returns its manifest entry with hashes, sizes, rows and columns
def download(name, store, force=False):
    spec = NATURAL_DATASETS[name]
    folder = natural_cache_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for file_name, file_spec in spec["files"].items():
        path = folder / file_name
        if force or not path.exists():
            path.write_bytes(file_bytes(file_spec, store))
        data = path.read_bytes()
        files[file_name] = dict(file_spec, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if file_name.endswith(".csv"):
            table = pd.read_csv(path, sep=file_spec.get("sep", ","), dtype=str, keep_default_na=False, encoding="utf-8-sig", low_memory=False)
            files[file_name].update(rows=len(table), columns=table.shape[1])
    entry = {key: value for key, value in spec.items() if key != "files"}
    return dict(entry, files=files, checked=datetime.date.today().isoformat())

# downloads the datasets and writes benchmark/results/natural.json
def build_manifest(names=None, force=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    store = {}
    for name in names or list(NATURAL_DATASETS):
        manifest[name] = download(name, store, force)
        shapes = ", ".join(f"{file} {entry.get('rows', '-')}x{entry.get('columns', '-')}" for file, entry in manifest[name]["files"].items())
        print(f"{name} ({manifest[name]['kind']}): {shapes}", flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest

# command line: python -m benchmark.natural [names...] [--force]
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Download the naturally messy and leaky datasets.")
    parser.add_argument("names", nargs="*")
    parser.add_argument("--force", action="store_true", help="download again even if the files exist")
    arguments = parser.parse_args()
    build_manifest(arguments.names or None, arguments.force)
    return 0

if __name__ == "__main__":
    sys.exit(main())
