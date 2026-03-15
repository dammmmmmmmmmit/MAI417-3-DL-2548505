# MAI417-3 — Deep Learning Lab Portfolio

**Course**: MAI417-3 | Deep Learning  
**Student ID**: 2548505

A collection of 7 deep learning lab assignments covering foundational to advanced neural network architectures. Each lab folder contains the Jupyter notebook and a detailed README.

---

## Lab Overview

### Lab 01 — XOR Classification with MLP
Implemented a **Multi-Layer Perceptron** to solve the XOR problem — a classic non-linear classification task. Built twice using both **Keras/TensorFlow** and **PyTorch** to compare framework behaviours. Achieved 100% classification accuracy after 1,000 training epochs.

**Skills**: `TensorFlow` · `Keras` · `PyTorch` · `Neural Networks` · `Binary Classification` · `Adam Optimizer` · `Non-linear Activation Functions`

---

### Lab 02 — FashionMNIST Image Classification (CNN)
Built a **Convolutional Neural Network (CNN)** in PyTorch to classify 70,000 clothing images across 10 categories. Leveraged GPU acceleration and standard data augmentation/normalisation pipelines.

**Skills**: `PyTorch` · `Convolutional Neural Networks (CNN)` · `Image Classification` · `Computer Vision` · `GPU/CUDA` · `Transfer Learning Fundamentals` · `torchvision`

---

### Lab 03 — Diabetes Prediction with Regularised Classifiers
Applied **Logistic Regression, Lasso (L1), Ridge (L2), and ElasticNet** classifiers to predict diabetes onset from clinical features. Demonstrates understanding of regularisation, overfitting prevention, and scikit-learn pipelines.

**Skills**: `scikit-learn` · `Logistic Regression` · `L1/L2 Regularisation` · `Feature Engineering` · `Classification Metrics` · `Data Preprocessing` · `pandas` · `NumPy`

---

### Lab 04 — Object Detection on VisDrone Dataset (YOLO + Custom CNN)
Implemented **real-time object detection** on drone-captured imagery. Used pretrained **YOLOv5 and YOLOv8** (transfer learning) alongside a custom from-scratch CNN with **CuPy GPU acceleration**. Handled YOLO annotation format and multi-class bounding box detection.

**Skills**: `YOLOv5` · `YOLOv8` · `Object Detection` · `Transfer Learning` · `CuPy` · `CUDA GPU` · `Custom Neural Network` · `Bounding Box Regression` · `Ultralytics` · `Computer Vision`

---

### Lab 05 — Character-Level Text Generation (LSTM/RNN)
Built a **character-level language model** using LSTM/RNN that learns text patterns and generates new text in the style of the training corpus. Demonstrates sequence modelling, one-hot encoding, and autoregressive generation.

**Skills**: `LSTM` · `RNN` · `Sequence Modelling` · `Natural Language Processing (NLP)` · `Text Generation` · `One-Hot Encoding` · `Generative Models` · `PyTorch/TensorFlow`

---

### Lab 06 — Sparse Autoencoder on CIFAR-10
Implemented a **Sparse Autoencoder** that compresses 3,072-pixel colour images into a 512-dimensional bottleneck representation. Combined MSE reconstruction loss with L1 sparsity regularisation, achieving 69% loss reduction over 20 epochs.

**Skills**: `PyTorch` · `Autoencoders` · `Unsupervised Learning` · `Representation Learning` · `Sparse Regularisation (L1)` · `Image Compression` · `CUDA GPU` · `CIFAR-10`

---

### Lab 07 — Variational Autoencoder (VAE) for Image Generation
Built a **Variational Autoencoder (VAE)** that learns a probabilistic latent space over MNIST digits, enabling both reconstruction and **novel image generation**. Implements the reparameterisation trick and KL divergence regularisation.

**Skills**: `PyTorch` · `Variational Autoencoder (VAE)` · `Generative AI` · `Probabilistic Deep Learning` · `KL Divergence` · `Latent Space` · `Reparameterisation Trick` · `MNIST` · `CUDA GPU`

---

## Tech Stack

| Category               | Tools & Libraries                                           |
|------------------------|-------------------------------------------------------------|
| Deep Learning          | PyTorch, TensorFlow, Keras                                  |
| Computer Vision        | YOLOv5, YOLOv8, Ultralytics, torchvision, CNN              |
| NLP / Seq. Modelling   | LSTM, RNN, character-level language models                  |
| Generative Models      | VAE, Sparse Autoencoder                                     |
| GPU Computing          | CUDA, CuPy                                                  |
| Classical ML           | scikit-learn, Logistic Regression, Regularisation           |
| Data & Visualisation   | pandas, NumPy, Matplotlib, Seaborn                          |
| Language               | Python 3                                                    |
