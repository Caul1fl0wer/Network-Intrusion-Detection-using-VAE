# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


COLUMN_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]


def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def build_preprocessor(df):
    categorical = ["protocol_type", "service", "flag"]
    numerical = [col for col in df.columns if col not in categorical + ["label", "difficulty"]]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    return preprocessor


def preprocess_train(df):
    df_normal = df[df["label"] == "normal"]

    X = df_normal.drop(columns=["label", "difficulty"])

    preprocessor = build_preprocessor(df)
    X_processed = preprocessor.fit_transform(X)

    return X_processed, preprocessor


def preprocess_test(df, preprocessor):
    y = df["label"].values
    X = df.drop(columns=["label", "difficulty"])

    X_processed = preprocessor.transform(X)

    return X_processed, y
