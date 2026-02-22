# src/train.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm

from preprocessing import load_data, preprocess_train
from model import VAE

import joblib


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
LATENT_DIM = 16


def loss_function(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl


def main():

    df = load_data("../data/KDDTrain+.txt")
    X_train, preprocessor = preprocess_train(df)

    X_train = torch.tensor(X_train, dtype=torch.float32)

    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train.shape[1]

    model = VAE(input_dim, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(loader):
            x = batch[0].to(DEVICE)

            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss = loss_function(recon, x, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

    torch.save(model.state_dict(), "../vae_model.pth")
    # after preprocessing and training
    joblib.dump(preprocessor, "../preprocessor.pkl")

if __name__ == "__main__":
    main()
