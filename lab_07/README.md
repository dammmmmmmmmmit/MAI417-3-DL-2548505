# Lab 07 — Variational Autoencoder (VAE) for Image Generation (MNIST)

## About This Program

This lab implements a **Variational Autoencoder (VAE)** using **PyTorch** — a generative deep learning model that learns a *probabilistic latent space* from handwritten digit images (MNIST). Unlike a standard autoencoder, the VAE can **generate entirely new images** by sampling from the learned latent distribution.

VAEs are foundational to modern generative AI and are a stepping stone to more advanced models like diffusion models and GANs.

---

## Dataset

| Property         | Detail                                          |
|------------------|-------------------------------------------------|
| Dataset          | MNIST (torchvision, auto-downloaded)            |
| Training samples | 60,000 handwritten digit images                 |
| Image size       | 28 × 28 pixels, grayscale = 784 pixels          |
| Classes          | Digits 0–9 (not used during training — unsupervised) |

**Preprocessing**: Normalised to `[-1, 1]` range (mean=0.5, std=0.5).

---

## Model Architecture

```
Encoder:
  Input:      784 (flattened 28×28 image)
  fc1:        784 → 400  + ReLU
  fc_mu:      400 → 20   (outputs mean μ of latent distribution)
  fc_var:     400 → 20   (outputs log variance log(σ²) of latent distribution)

Reparameterisation Trick:
  z = μ + ε × σ   (ε ~ N(0,I)) — allows gradients to flow through sampling

Decoder:
  Input:      20 (latent vector z)
  fc1:        20  → 400  + ReLU
  fc_out:     400 → 784  + Tanh  (reconstructed image)
```

### Hyperparameters

| Parameter      | Value  |
|----------------|--------|
| Input Dim      | 784    |
| Hidden Dim     | 400    |
| Latent Dim     | 20     |
| Batch Size     | 128    |
| Epochs         | 10     |
| Learning Rate  | 1e-3   |

---

## Loss Function

```
VAE Loss = Reconstruction Loss (MSE) + KL Divergence Loss
```

- **Reconstruction Loss**: How closely the output image matches the input — directly measures image quality
- **KL Divergence**: Measures how far the learned latent distribution strays from a standard normal N(0,1) — acts as a regulariser to keep the latent space smooth and continuous

---

## Output Interpretation

Training losses across 10 epochs (averaged per sample):

```
Epoch [ 1/10]  Reconstruction: 98.66  |  KL Divergence: 22.34
Epoch [ 2/10]  Reconstruction: 55.33  |  KL Divergence: 26.80
Epoch [ 3/10]  Reconstruction: 49.52  |  KL Divergence: 27.42
Epoch [ 5/10]  Reconstruction: 44.87  |  KL Divergence: 27.83
Epoch [10/10]  Reconstruction: 40.79  |  KL Divergence: 28.14
```

**What this means**:
- **Reconstruction loss dropped 59%** (from 98.66 → 40.79) — the model rapidly learns to decode recognisable digits from the 20-dimensional latent space
- **KL divergence stabilises around 28** — the latent space is converging to a shape close to N(0,1), which is essential for smooth image generation
- **Final output**: A grid of 10 original vs. 10 reconstructed digit images shows the VAE produces visually similar reconstructions — blurry but structurally correct — which is typical for VAEs trained at this scale
- **Generation**: By sampling random vectors from N(0,1) and passing through the decoder, the model produces brand-new handwritten digits not seen during training

---

## Technologies Used

- Python 3
- PyTorch (`torch.nn`, `torch.optim`, `torchvision`)
- CUDA (GPU acceleration — `device: cuda`)
- Matplotlib (visualisation of original vs reconstructed images)
