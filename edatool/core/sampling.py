import numpy as np
import pandas as pd

# returns a random sample of a Series, or the whole Series if it is small
def sample_values(series, size, seed=0):
    if len(series) <= size:
        return series
    return series.sample(n=size, random_state=seed)

# returns the row ids of a sample; for classification the sample keeps the...
# class shares
def sample_index(index, size, strata=None, seed=0):
    if len(index) <= size:
        return index
    if strata is None:
        picked = np.random.default_rng(seed).choice(len(index), size=size, replace=False)
        return index[np.sort(picked)]
    groups = pd.Series(strata.astype(str).to_numpy(), index=index)
    frame = groups.to_frame("group")
    picked = frame.groupby("group", group_keys=False).sample(frac=size / len(index), random_state=seed)
    return picked.index.sort_values()
