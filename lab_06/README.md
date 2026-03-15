# Lab 06 — Sparse Autoencoder for Image Reconstruction (CIFAR-10)

## About This Program

This lab implements a **Sparse Autoencoder** using **PyTorch** to learn compressed, sparse representations of images from the **CIFAR-10 dataset**. An autoencoder learns to *encode* an image into a low-dimensional latent vector and then *decode* it back to reconstruct the original image — without any labels.

The **sparsity constraint** forces most neurons in the bottleneck layer to stay inactive (close to zero), encouraging the model to learn efficient, disentangled features.

---

## Dataset

| Property         | Detail                                                                     |
|------------------|----------------------------------------------------------------------------|
| Dataset          | CIFAR-10 (torchvision, auto-downloaded)                                    |
| Training samples | 50,000                                                                     |
| Test samples     | 10,000                                                                     |
| Image size       | 32 × 32 × 3 (RGB) = 3,072 pixels per image                               |
| Classes          | 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)        |

**Preprocessing**: Images normalised to `[-1, 1]` range per RGB channel (mean=0.5, std=0.5).

---

## Model Architecture

```
Input:      3,072 (flattened 32×32×3 image)

Encoder:
  Linear  3072 → 1024  + ReLU
  Linear  1024 → 512   + ReLU  ← Bottleneck (compressed representation)

Decoder:
  Linear   512 → 1024  + ReLU
  Linear  1024 → 3072  + Tanh  ← Reconstructed image (range: [-1, 1])
```

### Hyperparameters

| Parameter        | Value  |
|------------------|--------|
| Batch Size       | 128    |
| Learning Rate    | 1e-3   |
| Epochs           | 20     |
| Hidden Dim       | 512    |
| Sparsity Lambda  | 1e-4   |

---

## Loss Function

```
Total Loss = Reconstruction Loss (MSE) + λ × Sparsity Penalty (L1)
```

- **Reconstruction Loss (MSE)**: Measures how closely the decoded image matches the original — lower = better reconstruction
- **Sparsity Penalty (L1 norm)**: Penalises large activations in the bottleneck layer — encourages most neurons to be inactive (sparse)

---

## Output Interpretation

Training output across 20 epochs:

```
Epoch [01/20]  Total Loss: 0.06093  Recon: 0.03777  Sparsity: 0.51186
Epoch [05/20]  Total Loss: 0.02436  Recon: 0.02364  Sparsity: 0.59933
Epoch [10/20]  Total Loss: 0.02052  Recon: 0.01797  Sparsity: 0.57163
Epoch [15/20]  Total Loss: 0.01948  Recon: 0.02022  Sparsity: 0.56373
Epoch [20/20]  Total Loss: 0.01902  Recon: 0.01763  Sparsity: 0.52715
```

**What this means**:
- **Total loss dropped ~69%** (from 0.061 to 0.019) — the model is learning to reconstruct images much better over training
- **Reconstruction loss dropped from 0.038 → 0.018** — the decoded images are increasingly faithful to the originals
- **Sparsity values ~0.52–0.60** — roughly half the bottleneck neurons are active on average, confirming the sparse representation is being enforced
- The model successfully learns to compress 3,072 pixels into just 512 values and reconstruct recognisable images

---

## Technologies Used

- Python 3
- PyTorch (`torch.nn`, `torch.optim`, `torchvision`)
- CUDA (GPU acceleration — `device: cuda`)
- NumPy
- Matplotlib
