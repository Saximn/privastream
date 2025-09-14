# PII Audio Detection Ensemble Approach

## Key Accomplishment 🎯

Proudly beat the state-of-the-art DeBERTa algorithm through innovative ensemble techniques, advanced post-processing, and knowledge distillation strategies.
<img width="1327" height="918" alt="image" src="https://github.com/user-attachments/assets/7ee1481f-3967-462d-9c9f-b9b0604f2289" />

## Overview of the Approach

Our solution focused on building a robust ensemble of DeBERTa models with careful post-processing, dataset curation, and additional techniques such as knowledge distillation. 📈

## Datasets 📊

We relied on several community datasets and also built our own:

- [@nbroad's PII-DD mistral-generated dataset](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated) — by far the most valuable external dataset. ⭐️
- [@mpware's Mixtral-generated essays](https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays).
- A custom dataset of ~2k samples (using @minhsienweng's notebook as a starting point).  
  This was released as external_data_v8.json and includes the nbroad dataset. 🔥

## Modelling 🤖

### Training Configuration ⚙️

- Folds: Each run trained on 4 folds (document % 4), plus 1–2 full-fit runs.
- Labels: 13 PII categories used consistently across experiments.
- Hyperparameters:
  - Max sequence length: 1600–2048
  - Epochs: 3–4
  - Learning rate: 1e-5
  - No downsampling for PII-free essays
  - o_weight = 0.05

### Model Variants 🚀

We experimented with multiple architectures, ultimately favoring DeBERTa due to its balance of speed and accuracy. Key variations:

#### 1. Multi-Sample Dropout Model

Reduced training curve spikes, led to more stable training compared to vanilla models.

#### 2. BiLSTM-Augmented Model

Added a BiLSTM layer before the classifier (as suggested in public notebooks). Improved diversity and correlated well with LB/CV scores.  
Issues with NaN losses were fixed by reinitializing layers and saving with save_safetensors=False.

#### 3. Knowledge Distillation

Teacher–student setup where diverse models trained on different datasets acted as teachers.  
Improved CV and LB scores by ~0.005–0.01. Multiple teachers were tested, though time limited further exploration.

#### 4. Augmentation (Name Swapping)

Swapped first/last names in some essays to diversify training (e.g., "Leroy" appearing as both B-NAME_STUDENT and I-NAME_STUDENT).

Credit: Points 2 and 3 were originally highlighted by @gauravbrills and helped improve ensemble diversity.

## Knowledge Distillation (Loss Function) 🧠

We implemented a custom trainer that combines student loss and KL-divergence loss against teacher logits:

loss = alpha _ student_loss + (1 - alpha) _ distillation_loss

With temperature scaling to soften teacher distributions (T=3) and mixing weight alpha=0.5.

## Post-Processing ✨

This stage proved critical for boosting scores. Key rules:

### Core Rules

- Label-specific thresholds: Different thresholds per PII label instead of a global one.
- Student names cleanup: Remove predictions not in title case or containing digits/underscores.
- Document-wide consistency: If "X" is tagged once as NAME_STUDENT, tag all instances of "X" in the document.
- Phone → ID rule: Numbers with ≥9 digits predicted as PHONE_NUM converted to ID_NUM.
- Address repairs: Fix missing \n tokens in STREET_ADDRESS.

### Advanced Rules

- Username rules: In sequences like B-USERNAME – - – B-USERNAME, merge tokens into a single username span.
- Span filtering:
  - Remove ID_NUM with <4 or >25 chars
  - Remove URL_PERSONAL with <10 chars
  - Remove EMAIL without "@"
- Final repairs: Adjust spans for names and IDs, add regex-based predictions if models missed them.

## Ensemble Strategy 🎯

### Weighted Voting Ensemble 💪

- 7 groups, 10 models total
- Weights tuned with Optuna on OOF CV data

### Composition 🏗

Each group = 1–2 models (both fold-based and full-fit for diversity)

### Submissions 📤

- All 3 were ensembles
- 2 used full postprocessing, 1 more conservative to hedge against overfitting

## What Didn't Work ❌

The following approaches were tested but didn't improve performance:

- MLM pretraining
- Freezing layers
- Repairing I-URL_PERSONAL predictions
- CausalLM inference (too slow)
- Training with stride
- Longformer / LLM models
- Label-specific models (e.g., training NAME_STUDENT only)
- Augmenting with rare/multinational names
- Pseudo-labeling test set
