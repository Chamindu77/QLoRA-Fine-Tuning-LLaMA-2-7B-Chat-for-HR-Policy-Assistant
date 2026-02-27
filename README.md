# QLoRA Fine-Tuning: LLaMA-2-7B-Chat for HR Policy Assistant

**Domain:** Sri Lankan HR Policies and Labor Law  
**Model:** `meta-llama/Llama-2-7b-chat-hf`  
**Method:** QLoRA (4-bit NF4 Quantization + LoRA Adapters)  
**Author:** Chamindu Nipun  
**Date:** 27/02/2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Running the Notebook](#5-running-the-notebook)
6. [Model Architecture Summary](#6-model-architecture-summary)
7. [Hyperparameters](#7-hyperparameters)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation](#9-evaluation)
10. [Inference](#10-inference)
11. [Deployment](#11-deployment)
12. [Known Limitations](#12-known-limitations)
13. [References](#13-references)

---

## 1. Project Overview

This project implements a **domain-specific HR Policy Assistant** for Sri Lankan enterprises by fine-tuning `LLaMA-2-7B-Chat` using **QLoRA** (Quantized Low-Rank Adaptation).

The assistant accurately answers HR-related employee queries — such as leave entitlements, resignation notice periods, and maternity benefits — aligned with:

- Sri Lankan Labor Laws (Shop and Office Employees Act, Maternity Benefits Ordinance, Industrial Disputes Act)
- Company-specific HR policies and employee handbooks
- A formal, professional corporate tone

### Why Fine-Tuning Instead of Prompt Engineering?

| Approach | Consistency | Domain Accuracy | Hallucination Risk | Cost at Scale |
|----------|-------------|------------------|--------------------|---------------|
| Prompt Engineering (GPT-3.5) | Low | Low | High | High (per-token) |
| QLoRA Fine-Tuning (LLaMA-2) | High | High | Low | Zero (self-hosted) |

### Why LLaMA-2 over GPT-3.5 Turbo?

- **Open weights** — QLoRA fine-tuning requires full weight access. GPT-3.5 Turbo is API-only.
- **Data privacy** — Sensitive HR data never leaves the corporate network.
- **Zero inference cost** — No per-token API fees at production scale.
- **Full deployment control** — On-premise, cloud, or edge deployment options.

---

## 2. System Requirements

### Minimum (Training)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GPU with 16 GB VRAM (RTX 3090, RTX 4080, A4000) |
| RAM | 32 GB system RAM |
| Storage | 50 GB free disk space |
| CUDA | 11.8 or higher |
| Python | 3.9 or higher |
| OS | Ubuntu 20.04+ / Windows WSL2 |

### Recommended (Training)

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA A100 40 GB (cloud) or RTX 4090 24 GB (on-premise) |
| RAM | 64 GB system RAM |
| Storage | 100 GB SSD |

### Cloud GPU Options

| Provider | Instance | VRAM | Estimated Cost |
|----------|----------|------|----------------|
| Google Colab Pro+ | A100 | 40 GB | ~$10/month |
| AWS | p3.2xlarge (V100) | 16 GB | ~$3.06/hr |
| GCP | a2-highgpu-1g (A100) | 40 GB | ~$3.67/hr |
| RunPod | RTX 4090 | 24 GB | ~$0.74/hr |

---

## 3. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/qlora-hr-policy-assistant.git
cd qlora-hr-policy-assistant
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0
trl>=0.7.0
datasets>=2.14.0
accelerate>=0.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
evaluate>=0.4.0
rouge-score>=0.1.2
bert-score>=0.3.13
```

### Step 4 — Hugging Face authentication

LLaMA-2 requires acceptance of Meta's license on Hugging Face and authentication:

```bash
huggingface-cli login
# Enter your Hugging Face access token when prompted
# Request LLaMA-2 access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

---

## 4. Dataset Preparation

### Expected Format

Each JSON file contains a list of Q&A pair objects:

```json
[
  {
    "question": "How many casual leave days are employees entitled to per year?",
    "answer": "Under the Shop and Office Employees Act, employees are entitled to 7 days of casual leave per calendar year. Casual leave cannot be carried forward or encashed."
  },
  {
    "question": "What is the maternity leave entitlement under Sri Lankan law?",
    "answer": "Under the Maternity Benefits Ordinance, female employees are entitled to 84 days (12 weeks) of paid maternity leave for the first two live births, and 42 days for subsequent births."
  }
]
```

### Dataset Split

| Split | File | Samples | Purpose |
|-------|------|---------|---------|
| Train | `hr_dataset_train.json` | ~800 | Model weight optimization |
| Validation | `hr_dataset_val.json` | ~100 | Epoch-level loss monitoring |
| Test | `hr_dataset_test.json` | ~100 | Final held-out evaluation |

### Data Sources

- Sri Lankan Government labor law publications (Shop and Office Employees Act, Industrial Disputes Act, Wages Board Ordinances, Maternity Benefits Ordinance)
- Company HR policy handbooks and employee manuals
- Internal HR FAQ collections and ticketing system archives

### Data Quality Requirements

Before including any Q&A pair in the dataset, verify:

- Answer is accurate against the relevant Sri Lankan labor law clause
- Response uses formal, professional HR corporate tone
- No personally identifiable employee information is present
- No duplicate or near-duplicate pairs (cosine similarity threshold < 0.9)
- Text is clean, with no formatting artifacts or special characters

---

## 5. Running the Notebook

### Launch Jupyter

```bash
jupyter notebook qlora_hr_policy_assistant.ipynb
# or
jupyter lab qlora_hr_policy_assistant.ipynb
```

### Notebook Steps

| Step | Cell | Description |
|------|------|-------------|
| 1 | Install and Import | Install all packages, verify CUDA availability |
| 2 | Load Tokenizer | Load LLaMA-2 tokenizer, configure pad token |
| 3 | Quantization Config | Set up 4-bit NF4 BitsAndBytesConfig |
| 4 | Load Base Model | Load frozen 7B model with quantization |
| 5 | LoRA Config | Configure rank, alpha, target modules |
| 6 | Apply LoRA | Inject adapters, verify trainable parameter count |
| 7 | Load Dataset | Load JSON files, apply chat template formatting |
| 8 | Training Arguments | Configure optimizer, scheduler, batch size |
| 9 | SFTTrainer | Initialize supervised fine-tuning trainer |
| 10 | Run Training | Execute training loop, monitor loss |
| 11 | Save Adapters | Save LoRA adapter weights (~50 MB) |
| 12 | Merge Model | Fuse adapters into base model for deployment |
| 13 | Inference Test | Run sample HR queries and verify output |

### Run All Cells

```
Kernel > Restart & Run All
```

---

## 6. Model Architecture Summary

| Property | Value |
|----------|-------|
| Base Model | LLaMA-2-7B-Chat (Meta AI) |
| Architecture | Decoder-only Transformer |
| Parameters | 7 Billion (frozen) |
| Positional Encoding | Rotary Positional Embeddings (RoPE) |
| Attention | Grouped-Query Attention (GQA) |
| Normalization | Pre-normalization with RMSNorm |
| Activation | SwiGLU |
| Context Window | 4,096 tokens |
| Alignment | SFT + RLHF (PPO) |
| Quantization | 4-bit NF4 (BitsAndBytes QLoRA) |
| Trainable Params | ~524,288 (LoRA adapters only) |
| Trainable Ratio | < 0.01% of total parameters |

---

## 7. Hyperparameters

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| LoRA Rank (r) | 16 | Balances expressiveness and efficiency for narrow HR domain |
| LoRA Alpha | 32 | Standard scaling: alpha = 2 x rank |
| LoRA Dropout | 0.1 | Prevents overfitting on small dataset |
| Target Modules | q_proj, v_proj | Best quality-per-parameter trade-off |
| Learning Rate | 2e-4 | Standard QLoRA learning rate |
| LR Scheduler | Cosine + Warmup | 10% warmup, smooth cosine decay |
| Batch Size | 4 (per device) | Fits 16 GB VRAM with QLoRA |
| Gradient Accumulation | 4 steps | Effective batch size = 16 |
| Epochs | 3 | Sufficient for domain convergence |
| Max Sequence Length | 512 tokens | Covers all HR Q&A pairs |
| Optimizer | paged_adamw_8bit | Memory-efficient AdamW |
| Weight Decay | 0.01 | Regularization |
| Max Grad Norm | 0.3 | Gradient clipping |
| Precision | BF16 | Stable on Ampere+ GPUs |

---

## 8. Training Pipeline

```
Raw HR Data (Laws + Policies + FAQs)
            |
            v
    Data Collection & Cleaning
    (de-dup, anonymize, normalize)
            |
            v
    JSON Q&A Pair Formatting
    (LLaMA-2 chat template)
            |
            v
    80 / 10 / 10 Train/Val/Test Split
            |
            v
    Load LLaMA-2-7B-Chat (4-bit NF4)
    [Base model weights FROZEN]
            |
            v
    Inject LoRA Adapters
    (q_proj + v_proj, r=16, alpha=32)
            |
            v
    SFTTrainer (3 epochs)
    paged_adamw_8bit, lr=2e-4, BF16
            |
            v
    Validation Loss Monitoring
    (early stopping patience=2)
            |
            v
    Save LoRA Adapter Weights
            |
            v
    Merge Adapters into Base Model
            |
            v
    Evaluation (ROUGE + BERTScore + Human)
            |
            v
    Deploy via FastAPI + Docker
```

---

## 9. Evaluation

### Automated Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| ROUGE-1 | >= 0.55 | Unigram overlap with reference answers |
| ROUGE-2 | >= 0.35 | Bigram overlap |
| ROUGE-L | >= 0.45 | Longest common subsequence |
| BERTScore F1 | >= 0.82 | Semantic similarity via BERT embeddings |
| Perplexity (PPL) | < 15 | Text fluency and confidence |

### Human Evaluation

| Criterion | Target | Evaluator |
|-----------|--------|-----------|
| Policy Accuracy | >= 90% | Senior HR professional |
| Hallucination Rate | < 5% | Senior HR professional |
| Tone Compliance | >= 95% | Senior HR professional |

### Run Evaluation

```python
from evaluate import load

rouge = load("rouge")
results = rouge.compute(
    predictions=model_answers,
    references=reference_answers
)
print(results)
```

---

## 10. Inference

### Direct Python Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_PATH = './hr_llama2_merged'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map='auto'
)

SYSTEM_PROMPT = (
    'You are a professional HR Policy Assistant for a Sri Lankan company. '
    'Answer all queries based on Sri Lankan labor laws and company HR policies. '
    'Maintain a formal, professional tone.'
)

def ask_hr_assistant(question):
    prompt = (
        f'<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n'
        f'{question} [/INST]'
    )
    hr_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)
    output = hr_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        top_p=0.9
    )
    return output[0]['generated_text'].split('[/INST]')[-1].strip()

print(ask_hr_assistant('How many casual leave days are allowed per year?'))
```

### FastAPI Server

```bash
python serve.py
# API available at: http://localhost:8000
# Docs available at: http://localhost:8000/docs
```

**Sample API request:**

```bash
curl -X POST "http://localhost:8000/api/hr-query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the notice period for resignation?"}'
```

**Sample API response:**

```json
{
  "question": "What is the notice period for resignation?",
  "answer": "Under the Shop and Office Employees Act, the standard notice period for resignation is one month, unless otherwise specified in the employment contract. The employer may waive the notice period at their discretion."
}
```

---

## 11. Deployment

### Docker

```bash
# Build image
docker build -t hr-policy-assistant .

# Run container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name hr-assistant \
  hr-policy-assistant
```

### Docker Compose

```bash
docker-compose up -d
```

### CPU-Only Deployment (llama.cpp)

For environments without GPU, convert the model to GGUF format and run with llama.cpp:

```bash
# Convert to GGUF Q4_K_M quantization
python convert.py ./hr_llama2_merged --outtype q4_k_m

# Run with llama.cpp
./llama-cli -m hr_llama2_merged.Q4_K_M.gguf \
            -p "[INST] How many casual leave days are allowed? [/INST]" \
            -n 256
```

### Monitoring

| Tool | Purpose |
|------|---------|
| Prometheus | Latency, throughput, error rate metrics |
| Grafana | Dashboard visualization |
| ELK Stack | Query logging and drift detection |
| TensorBoard | Training loss and metric curves |

---

## 12. Known Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| Knowledge Cutoff | LLaMA-2 training data ends July 2023 | Dataset sourced from current official documents; periodic retraining planned |
| No Real-Time Access | Cannot fetch live policy updates | Future RAG layer will index latest labor law documents |
| GPU Required | Minimum 16 GB VRAM for training | QLoRA reduces requirement; cloud GPU options available |
| English Only | Limited Sinhala and Tamil capability | Phase 2 will include multilingual dataset extension |
| Context Window | 4,096 token limit | Long documents segmented into chunks during preprocessing |

---

## 13. References

- **QLoRA Paper:** Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs* — https://arxiv.org/abs/2305.14314
- **LLaMA-2 Paper:** Touvron et al. (2023) — *Llama 2: Open Foundation and Fine-Tuned Chat Models* — https://arxiv.org/abs/2307.09288
- **LoRA Paper:** Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models* — https://arxiv.org/abs/2106.09685
- **gpt-llm-trainer Repository:** https://github.com/mshumer/gpt-llm-trainer
- **Hugging Face PEFT:** https://github.com/huggingface/peft
- **TRL SFTTrainer:** https://github.com/huggingface/trl
- **BitsAndBytes:** https://github.com/TimDettmers/bitsandbytes
- **Sri Lankan Labor Law:** Department of Labour, Sri Lanka — https://www.labourdept.gov.lk

---


