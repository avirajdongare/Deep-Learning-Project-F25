# 🧮 MathVerifier — Llama-3 8B Fine-Tuning for Math Answer Verification


This project fine-tunes **Llama-3 8B** using **Supervised Fine-Tuning (SFT)** on the **Math Question Answer Verification** dataset (Hugging Face). 

The goal is to predict whether a given math answer is **correct** or **incorrect**, based on the *question*, *answer*, and optionally the *solution/explanation*.

---

## 🔍 Overview

In this competition, participants fine-tune an LLM to verify math answers.  
Your model receives a math question, an answer, and a reasoning explanation, and must output a boolean label:
```
is_correct ∈ {True, False}
```

The fine-tuned model learns logical and mathematical reasoning consistency from provided solutions and outputs whether the given answer is correct.

---
### Kaggle Competition

https://www.kaggle.com/competitions/dl-fall-25-kaggle-contest/overview

### 🏋️ Model Weights

You can download the fine-tuned model here:

[📦 Download MathVerifier_Llama3-8B_Weights](https://drive.google.com/drive/folders/1mmK1Fu3ch8xtvKJ6H9Qr-VS6d7NB_JKk?usp=sharing)

---

### Contributors

| Name              | Responsibilities                           | 
| ----------------- | -------------------------------------------|
| Aviraj Dongare    | Model Training, LoRa Configuration, Tuning | 
| Swathi Awasthi    | Inferencing and Evaluation                 | 
| Joshua Leeman     | Preparing Data and Documentation           | 

---
### 🧱 Notebook Structure

| Section | Description |
|----------|--------------|
| **1. Dataset Loading** | Loads the Hugging Face dataset `ad6398/nyu-dl-teach-maths-comp` for math question–answer verification. |
| **2. Preprocessing** | Cleans and formats data into instruction-style prompts combining `question`, `answer`, and `solution`. |
| **3. Model Setup** | Initializes the base `meta-llama/Meta-Llama-3-8B` model and tokenizer for fine-tuning. |
| **4. Fine-Tuning (SFT)** | Uses `trl.SFTTrainer` with supervised fine-tuning on formatted prompts. |
| **5. Evaluation** | Computes validation accuracy and compares predicted correctness with ground truth labels. |
| **6. Inference and Submission** | Generates predictions on the test set and produces a `submission.csv` file for leaderboard evaluation. |

---
### ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Model | Llama-3 8B |
| Trainer | TRL `SFTTrainer` |
| Epochs | 3 |
| Batch Size | 32 |
| Learning Rate | 1e-5 |
| Sequence Length | 2048 |
| Optimizer | AdamW |
| Evaluation Metric | Accuracy |


---
### 📐 Example

> **Question:** What is the radius of the circle inscribed in triangle (ABC) if (AB = 22), (AC = 12), and (BC = 14)?  
> **Given Answer:** 3.16227766016838  
> **Given Solution:** The circle is inscribed in a triangle, and we know the sides of the triangle. To use the inradius formula, we need the area of the triangle. Using Heron’s formula:
>
> ```python
> import math
> from sympy import *
> AB, AC, BC = 22, 12, 14
> s = (AB + AC + BC) / 2
> K = sqrt(s * (s - AB) * (s - AC) * (s - BC))
> print(K)  # 75.8946638440411
> r = K / s
> print(r)  # 3.16227766016838
> ```
>
> Using the inradius formula, the answer is \\( \boxed{3.16227766016838} \\).
> 
> **is_correct = True**

--- 

### 📊 Dataset Description
The dataset comes from the **Math Question Answer Verification Competition** hosted on **Hugging Face**.  

It contains problems from multiple math domains and is structured as follows:

| Column | Description |
|:--------|:-------------|
| **question** | The math question posed to the student. |
| **answer** | The proposed or “ideal” answer to the question. |
| **solution** | A detailed reasoning or explanation that justifies the answer. |
| **is_correct** | Boolean label — `True` if the answer is correct, `False` otherwise. |

> **Note:** In the **test set**, all `is_correct` values are placeholders set to `True`.  
> The model must still generate predictions for evaluation.

### 🧩 Dataset Loading
```python
from datasets import load_dataset

dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
```
---

### 🧠 Task

Training a model that predicts the label is_correct from the fields: question, answer, and optionally solution.

- Build a supervised fine-tuning pipeline using Llama-3 8B (or an equivalent model).
- Leverage solution text as auxiliary supervision to improve reasoning.
- The output should be a boolean prediction per sample.

### ⚙️ Training Objective
  
This project applies Supervised Fine-Tuning (SFT) with the following setup:

- Base model: **Llama-3 8B**
- Framework: **Hugging Face / TRL (SFTTrainer)**
- Evaluation metric: **Accuracy on is_correct**
- Goal: **Learn reasoning consistency and logical correctness detection**

The fine-tuned model achieves ≈ 86 % accuracy on the validation set.

---

### 📂 Submission Format

- A file named submission.csv with the following structure:
```
is_correct
True
False
True
```
The rows correspond exactly to the order of the test dataset (test.csv).

### 🧾 Evaluation

- Submissions are evaluated using Accuracy on the is_correct column.

---

### 📈 Key Result

| Metric | Value |
|:--------|:-------|
| **Validation Accuracy** | **~86%** |
| **Model Size** | 8 Billion parameters |
| **Framework** | Hugging Face TRL (PEFT + LoRA) |










