# Lab 01 — XOR Classification with Multi-Layer Perceptron (MLP)

## About This Program

This lab implements a **Multi-Layer Perceptron (MLP)** to solve the classic **XOR classification problem** — a problem that cannot be solved by a simple linear model, making it a fundamental exercise in understanding **non-linear neural networks**.

The XOR (Exclusive OR) function takes two binary inputs and outputs `1` only when the inputs differ:

| Input A | Input B | XOR Output |
|---------|---------|------------|
| 0       | 0       | 0          |
| 0       | 1       | 1          |
| 1       | 0       | 1          |
| 1       | 1       | 0          |

The same network is implemented **twice** — once using **Keras/TensorFlow** and once using **PyTorch** — to compare both frameworks.

---

## Model Architecture

Both implementations use the same architecture:

```
Input Layer:   2 neurons (binary inputs A and B)
Hidden Layer:  4 neurons — activation: tanh (learns non-linear patterns)
Output Layer:  1 neuron  — activation: sigmoid (outputs probability 0–1)
Total params:  17
```

- **Loss Function**: Binary Cross-Entropy (standard for binary classification)
- **Optimizer**: Adam (adaptive learning rate)
- **Epochs**: 1000
- **Batch Size**: 4 (full dataset per batch)

---

## Output Interpretation

### Keras/TensorFlow Result:
```
final loss:     0.2617
final accuracy: 1.0000
```
The model achieved **100% accuracy** after 1000 training epochs. Although the loss (0.26) is not zero, the probabilities are correctly classified above/below the 0.5 threshold:

```
Input: [0. 0.] → predicted: 0.1093 → class: 0 → actual: 0  ✓
Input: [0. 1.] → predicted: 0.6834 → class: 1 → actual: 1  ✓
Input: [1. 0.] → predicted: 0.8169 → class: 1 → actual: 1  ✓
Input: [1. 1.] → predicted: 0.2928 → class: 0 → actual: 0  ✓
```

### PyTorch Result:
The PyTorch model converges more tightly, with loss dropping from `0.0564` (epoch 200) to just `0.0026` (epoch 1000), showing strong convergence.

Both frameworks successfully learn the XOR pattern — proving the MLP can capture non-linear decision boundaries.

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- PyTorch (`torch.nn`, `torch.optim`)
- NumPy
- Matplotlib
