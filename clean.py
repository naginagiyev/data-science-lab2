import os
import shutil
from paths import sandbox_dir, config_dir, data_dir, models_dir

for item in os.listdir(sandbox_dir):
    item_path = os.path.join(sandbox_dir, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

for item in os.listdir(models_dir):
    item_path = os.path.join(models_dir, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

keywords = ["train", "test", "val", "processed"]
for item in os.listdir(data_dir):
    if any(keyword in item for keyword in keywords):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

for filename in ["configuration.json"]:
    file_path = config_dir / filename
    if file_path.exists():
        os.remove(file_path)
