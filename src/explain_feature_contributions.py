# src/explain_feature_contributions.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from preprocessing import load_data, preprocess_test
from model import VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 16


def clean_feature_name(name):
    if "__" in name:
        return name.split("__")[1]
    return name


def base_feature(name):
    if name.startswith("protocol_type"):
        return "protocol_type"
    if name.startswith("service"):
        return "service"
    if name.startswith("flag"):
        return "flag"
    return name


def main(): # loading -> forward pass ->

    preprocessor = joblib.load("../preprocessor.pkl")

    df_test = load_data("../data/KDDTest+.txt")
    X_test, y_test = preprocess_test(df_test, preprocessor)

    feature_names = preprocessor.get_feature_names_out()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)


    input_dim = X_test.shape[1]

    model = VAE(input_dim, LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load("../vae_model.pth", map_location=DEVICE))
    model.eval()


    with torch.no_grad():
        recon, mu, logvar = model(X_test)
        feature_error = (X_test - recon) ** 2
        recon_error = torch.mean(feature_error, dim=1)

    # Select most anomalous sample
    idx = torch.argmax(recon_error).item()

    print("\n====================================")
    print("Most anomalous sample index:", idx)
    print("True label:", y_test[idx])
    print("Total anomaly score (mean MSE):", recon_error[idx].item())
    print("====================================\n")

    sample_feature_error = feature_error[idx].cpu().numpy()

    # Build explanation dataframe
    df_explain = pd.DataFrame({
        "feature_raw": feature_names,
        "squared_error": sample_feature_error
    })

    df_explain["clean_feature"] = df_explain["feature_raw"].apply(clean_feature_name)
    df_explain["base_feature"] = df_explain["clean_feature"].apply(base_feature)

    # Group contributions to make it cleaner
    grouped = (
        df_explain
        .groupby("base_feature")["squared_error"]
        .sum()
        .sort_values(ascending=False)
    )

    total_error = grouped.sum()

    grouped_df = pd.DataFrame({
        "total_squared_error": grouped,
        "percentage_contribution": 100 * grouped / total_error
    })

    print("Top 10 grouped feature contributions:\n")
    print(grouped_df.head(10))

    # Visualization
    top_grouped = grouped_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.bar(top_grouped.index, top_grouped["percentage_contribution"])
    plt.title("Feature Contribution to Anomaly Score (%)")
    plt.ylabel("Percentage Contribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()