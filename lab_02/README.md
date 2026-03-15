# Lab 02 — Fashion MNIST Image Classification with CNN (PyTorch)

## About This Program

This lab builds a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **Fashion MNIST dataset** — a benchmark dataset of 70,000 grayscale images across 10 clothing categories.

The goal is to correctly identify items like T-shirts, shoes, bags, coats, etc., from 28×28 pixel images using deep learning.

---

## Dataset

| Property        | Detail                                                      |
|-----------------|-------------------------------------------------------------|
| Dataset         | Fashion MNIST (torchvision)                                 |
| Training images | 60,000                                                      |
| Test images     | 10,000                                                      |
| Image size      | 28 × 28 pixels, grayscale                                  |
| Classes         | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |

**Preprocessing**: Images are converted to tensors and normalized (`mean=0.5, std=0.5`) so pixel values fall in the `[-1, 1]` range.

---

## Model Architecture (CNN)

```
Input:          (1, 28, 28) — grayscale image
Conv Layer 1:   Convolution + ReLU + MaxPooling
Conv Layer 2:   Convolution + ReLU + MaxPooling
Flatten
Fully Connected 1: Linear → ReLU
Fully Connected 2: Linear → 10 output classes
Output: Softmax probabilities across 10 categories
```

- **Loss Function**: Cross-Entropy Loss (multi-class classification)
- **Optimizer**: Adam
- **GPU accelerated**: Automatically uses CUDA if available

---

## Output Interpretation

The model trains over multiple epochs and outputs training/validation **loss** and **accuracy** curves.

- **Decreasing training loss** confirms the model is correctly learning patterns from clothing images.
- **High accuracy** on the test set shows the CNN generalises well to unseen data.
- CNNs outperform simple fully-connected networks on image data because they detect **local spatial features** (edges, textures, shapes) through convolution — exactly what clothing classification requires.

The sample images displayed at the start of training give a visual sense of the 10 diverse categories the model must distinguish.

---

## Technologies Used

- Python 3
- PyTorch (`torch`, `torch.nn`, `torchvision`)
- CUDA (GPU acceleration)
- NumPy
- Matplotlib
