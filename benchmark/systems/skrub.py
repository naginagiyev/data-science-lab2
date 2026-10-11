import sys
import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from benchmark.systems.common import run_adapter

# baseline "dirty data without an LLM": skrub's tabular pipeline...
# (TableVectorizer + HistGradientBoosting) on pd.read_csv input
def fit_predict(arguments):
    train = pd.read_csv(arguments.train)
    target = train.pop(arguments.target)
    tests = [pd.read_csv(path)[train.columns] for path in arguments.tests]
    if arguments.task == "regression":
        model = skrub.tabular_pipeline(HistGradientBoostingRegressor(random_state=arguments.seed), n_jobs=4).fit(train, target)
        return {"predictions": [model.predict(test) for test in tests]}
    model = skrub.tabular_pipeline(HistGradientBoostingClassifier(random_state=arguments.seed), n_jobs=4).fit(train, target)
    return {"predictions": [model.predict_proba(test) for test in tests], "classes": list(model.classes_)}

if __name__ == "__main__":
    sys.exit(run_adapter(fit_predict))
