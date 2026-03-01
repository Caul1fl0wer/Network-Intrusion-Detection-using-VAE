# Network Intrusion Detection using VAE

OBJECTIVE: 

This project focuses on building a Variational AutoEncoder (VAE) trained on benign network traffic to detect anomalous behavior.  

Here is the idea:

During training, the model learns a probabilistic distribution of healthy traffic. If, during evaluation phase, the model cannot reconstruct the sample with low reconstruction error, it's likely to be abnormal.

Beyond detection, this lab also explores:
- Latent space visualisation using t-SNE projection
- Feature-level contribution to reconstruction error for interpretability

NSL-KDD dataset (<href>https://github.com/HoaNP/NSL-KDD-DataSet</href>) has been used through this lab.

# Model architecture
**Encoder**
The encoder maps input data x to a Gaussian distribution parameterized by:

- μ = hWμ​+bμ
- log(σ2) = hWσ​+bσ

This mapping will shape our latent space. 

--

**Latent space**

Samples need to be differentiable to use gradient descent, which allows backpropagation during training on encoder weights. However, the encoder doesn't output a vector but a probability distribution instead such as z ~ N(μ,σ^2). The sample used during forward pass is not differentiable. Indeed, when z = f(μ,σ), we cannot compute ∂z/∂μ or ∂z/∂σ beacause the sampling process is no represented as a smooth function of μ and σ.

A solution to that problem is the reparametrization trick:

 - z = μ + σ⋅ϵ where z represent latent vector and ϵ ∼ N(0,1)

It's now differentiable as ∂z/∂μ ​= 1 and ∂z/∂σ ​= ϵ

The learned latent manifold is hence obtained by the encoder, which can be visualized using t-SNE:


<img width="975" height="731" alt="image" src="https://github.com/user-attachments/assets/4892b80e-cadb-4264-961f-3963e59ff81a" />


--

**Training Objective**

The training objective is to minimize the loss. In VAE, loss = -ELBO (Evidence Lower Bound), so we try to maximize +ELBO such as:
  - Loss = Reconstruction term + Regularization term
         = Mean Square Error (MSE) + KL-Divergence
    where MSE = (x - x')^2
          KL-divergence avoids exploding 𝜇 and collapsing σ by pushing 𝜇 to 0 and σ^2 to 1

--

**Anomaly Detection**
An anomaly score is computed to get a general insight on the sample as follows:
  - Reconstruction Error=1/n * (​∑(xj−x'​)^2)
  - Low error → sample lies near the learned manifold (normal traffic)
  - High error → sample lies outside the distribution (potential intrusion)

Evaluation is performed using ROC-AUC, measuring how well reconstruction error separates benign from malicious traffic.

# Feature contribution

To improve interpretability, feature-wise reconstruction error is computed:

This allows identification of which features contribute most to the anomaly score.

In our results:

- The feature xsnoop contributes the most.

- It is associated with R2L (Remote-to-Local) attacks.

- Top contributing features are related to privilege escalation.

This indicates that:

- Benign traffic rarely exhibits privilege escalation behavior.
- Large deviations in those features signal anomalous activity.

<img width="975" height="747" alt="image" src="https://github.com/user-attachments/assets/81d8911a-3cbc-4557-bc5e-e702d6d999bb" />


A histogram of top contributing features provides visual insight into anomaly structure.

<img width="975" height="578" alt="image" src="https://github.com/user-attachments/assets/45921189-7b80-41c5-95ff-a4705d669e63" />

