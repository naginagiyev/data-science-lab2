import json
import pandas as pd
from paths import config_dir, data_dir
from sklearn.model_selection import train_test_split

def split():
    with open(config_dir / "constants.json", 'r') as f:
        constants = json.load(f)

    configuration_path = config_dir / "configuration.json"
    with open(configuration_path, 'r') as f:
        runtime_config = json.load(f)

    train_split = constants["trainSplit"]
    test_split = constants["testSplit"]
    validation_split = 1 - train_split - test_split
    seed = constants["seed"]

    processed_data_path = runtime_config["processedDataFile"]
    processed_data = pd.read_csv(data_dir / processed_data_path)

    train, test = train_test_split(processed_data, test_size=test_split, random_state=seed)

    if validation_split > 0:
        adjusted_val_split = validation_split / (1 - test_split)
        train, val = train_test_split(train, test_size=adjusted_val_split, random_state=seed)
        val.to_csv(data_dir / "val.csv", index=False)
        runtime_config['valFile'] = "val.csv"

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    runtime_config['trainFile'] = "train.csv"
    runtime_config['testFile'] = "test.csv"

    with open(configuration_path, 'w') as f:
        json.dump(runtime_config, f, indent=2)
