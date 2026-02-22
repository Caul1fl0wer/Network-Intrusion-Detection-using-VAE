from preprocessing import load_data, preprocess_test
import torch
from model import VAE
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


preprocessor = joblib.load("../preprocessor.pkl")
df_test = load_data("../data/KDDTest+.txt")
X_test, y_test = preprocess_test(df_test, preprocessor)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

input_dim = X_test.shape[1]
latent_dim = 16

model = VAE(input_dim, latent_dim).to(DEVICE)
model.load_state_dict(torch.load("../vae_model.pth", map_location=DEVICE))
model.eval()

# Anomaly score - Reconstruction error
with torch.no_grad():
    recon, mu, logvar = model(X_test)
    recon_error = torch.mean((X_test - recon)**2, dim=1)  # per sample

# ELBO
def elbo(x, recon, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return -(recon_loss + kl)  # higher ELBO = more likely normal

with torch.no_grad():
    score = elbo(X_test, recon, mu, logvar)


labels = (y_test != "normal").astype(int)  # 1 = attack, 0 = normal
roc = roc_auc_score(labels, recon_error.cpu())
print("ROC-AUC:", roc)


z = mu.cpu().numpy()
z_2d = TSNE(n_components=2).fit_transform(z)
plt.scatter(z_2d[:,0], z_2d[:,1], c=(y_test!="normal"))
plt.show()

# need to wait a bit to get the plot