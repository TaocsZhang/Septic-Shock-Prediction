import numpy as np
import math
import pickle
import pandas as pd


def windowed_slice(x, y, seq_len):
    slices = []
    labels = []
    for i in range(x.shape[0] - (seq_len - 1)):
        s = x[i:i + seq_len, :]
        if y is not None:
            l = y[i:i + seq_len, :]
            labels.append(l)
        slices.append(s)
    return np.array(slices), np.array(labels)


def slices(x, y, seq_len):
    slices = []
    labels = []
    steps = math.ceil(x.shape[0] / seq_len)
    for i in range(steps):
        if i == steps - 1:
            s = x[i * seq_len:, :]
            l = y[i * seq_len:, :]
            s, l, _ = add_padding(s, l, seq_len)
            labels.append(l)
            slices.append(s)
        else:
            s = x[i * seq_len:i * seq_len + seq_len, :]
            l = y[i * seq_len:i * seq_len + seq_len, :]
            s = np.expand_dims(s, axis=0)
            l = np.expand_dims(l, axis=0)
            labels.append(l)
            slices.append(s)

    x = np.concatenate(slices, axis=0)
    y = np.concatenate(labels, axis=0)
    return x, y


def add_padding(x, y, seq_len, pad_value=-1):
    padding = seq_len - x.shape[0]
    x = np.pad(x, ((0, seq_len - x.shape[0]), (0, 0)), 'constant',
               constant_values=((pad_value, pad_value), (pad_value, pad_value)))
    x = np.expand_dims(x, axis=0)
    if y is not None:
        y = np.pad(y, ((0, seq_len - y.shape[0]), (0, 0)), 'constant',
                   constant_values=((pad_value, pad_value), (pad_value, pad_value)))
        y = np.expand_dims(y, axis=0)

    return x, y, padding


def process_dataframe(df, feature_cols, label_cols, seq_len, pad_value=-1):
    x = df[feature_cols].to_numpy()
    if label_cols is None:
        y = None
    else:
        y = df[label_cols].astype(int).to_numpy().flatten()
        y = np.array([y, 1 - y]).T
    features = []
    labels = []
    if x.shape[0] < seq_len:
        x, y, padding = add_padding(x, y, seq_len, pad_value)
    else:
        x = np.expand_dims(x[-seq_len:], axis=0)
        y = np.expand_dims(y[-seq_len:], axis=0)
        padding = 0
    return np.array(x), np.array(y), padding


def no_padding(df, feature_cols, label_cols):
    x = df[feature_cols].to_numpy()
    y = df[label_cols].astype(int).to_numpy().flatten()
    if x.shape[0] > seq_len:
        x = np.expand_dims(x[-seq_len:], axis=0)
        y = np.expand_dims(y[-seq_len:], axis=0)
    return np.array(x), np.array(y)

from tqdm import tqdm

def format_labels(df):
    if df["septic_shock"].any():
        df = df.reset_index(drop=True)
        df = df.iloc[:int(df["septic_shock_onset"].iloc[0])]
    return df

def zscore(df, means, std):
    return (df[means.keys()] - means) / std

def one_hot_encode(df, col, categories=None):
    if categories is not None:
        df[col] = df[col].astype(pd.CategoricalDtype(categories=categories, ordered=True))
    dummies = pd.get_dummies(df[col])
    df = pd.concat([df, dummies], axis=1)
    return df

def process_for_inference(df, means, stds, feature_cols, timesteps=2000):
    try:
        df = df.sort_values("chart_time")
        df = format_labels(df)
        df[list(means.keys())] = zscore(df, means, stds)
        df = one_hot_encode(df, "gender", categories=["M", "F"])
        df["age_bins"] = pd.cut(df["age"], [14, 20, 30, 40, 50, 70, 90])
        df = one_hot_encode(df, "age_bins")
        df.columns = [str(col) for col in df.columns]
        feature_columns = list(feature_cols + ["M", "F", "(14, 20]", "(20, 30]", "(30, 40]", "(40, 50]", "(50, 70]", "(70, 90]"])
        X, y, padding = process_dataframe(df, feature_columns, "septic_shock", timesteps)
        return X, y[:, :, 0], feature_columns, padding
    except Exception as e:
        print(f"Error for icu {df.iloc['patientid']}: {e}")
        
def load_feature_stats(feature_cols, version="0.0.1"):
    with open(f"data/stats/feature_stats_{version}.pkl", "rb") as f:
        feature_stats = pickle.load(f)
        
    means = {}
    std = {}
    for key in feature_stats.keys():
        if key in feature_cols:
            means[key] = feature_stats[key]["mean"]
            std[key] = feature_stats[key]["std"]
    return means, std

def get_dataframe(fh, patientid, DATA_SET_TYPE, DATA_VERSION):
    df = fh.get_object(f"preprocessed/{DATA_SET_TYPE}/{patientid}_{DATA_VERSION}.csv")
    return df