# Lab 05 — Character-Level Text Generation with LSTM/RNN

## About This Program

This lab implements a **character-level language model** using **Recurrent Neural Networks (RNN) / Long Short-Term Memory (LSTM)** networks. The model learns the statistical patterns of characters in a custom text file and can generate new, never-seen text that mimics the writing style of the training corpus.

Unlike word-level models, character-level models operate on individual characters — learning to form words, punctuation, and sentence structure purely from character sequences.

---

## Dataset

| Property      | Detail                                        |
|---------------|-----------------------------------------------|
| Input file    | `training_text.txt` (~28 KB custom text)      |
| Vocabulary    | All unique characters found in the text file  |
| Encoding      | One-hot encoding per character                |

---

## How It Works

1. **Text Preprocessing**
   - Text converted to lowercase for consistency
   - A vocabulary (set of all unique characters) is created
   - Each character is mapped to an integer index and vice versa

2. **Sequence Generation (Sliding Window)**
   - Input sequences of fixed length are created by sliding a window across the text
   - For each input sequence, the next character is the target output
   - Example: `"hello wor"` → predicts `"l"`

3. **One-Hot Encoding**
   - Each character converted to a binary vector of length = vocabulary size
   - e.g., if vocab has 50 chars, each character becomes a 50-dimensional vector

4. **Model**
   - LSTM/RNN layers process the sequence one character at a time
   - Final output layer uses Softmax to output a probability distribution over all characters
   - The next character is sampled from this distribution (with a "temperature" parameter controlling creativity)

---

## Output Interpretation

The generated text output starts to exhibit real patterns after sufficient training:

- **Early epochs**: Output is mostly random characters with little structure
- **Mid training**: Words start forming, spacing appears, sentence patterns emerge
- **Late training**: The model generates coherent phrases that stylistically match the training text

For example, if the training text is literary prose, the generated output will develop similar vocabulary and sentence flow. The model never memorises exact sentences — it learns the *statistical structure* of the language.

**Temperature parameter**: Lower temperature → more conservative/predictable output; higher temperature → more creative/random output.

---

## Technologies Used

- Python 3
- PyTorch or TensorFlow/Keras (LSTM/RNN layers)
- NumPy (one-hot encoding, sequence preparation)
- Custom text file (`training_text.txt`)
