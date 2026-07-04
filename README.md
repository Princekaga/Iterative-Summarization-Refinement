# Iterative Summarization Refinement via Log-Likelihood Optimization

> A reference-free NLP framework for evaluating and autonomously repairing LLM-generated text summaries using perplexity-guided iterative refinement — targeting hallucination mitigation without any ground-truth reference data.

---

## 🧑‍💻 Project Info

| | |
|---|---|
| **Organization** | Bharat Nirman Labs, IIT Kanpur |
| **Mentors** | Gurmeet · Pushpendra |
| **Duration** | May 2026 – Present |
| **Track** | NLP / Large Language Models / Hallucination Detection |
| **Target Venue** | ACL 2026 Student Research Workshop |

---

## 📌 Motivation

Traditional summarization metrics like **ROUGE** and **BLEU** require ground-truth reference summaries to evaluate quality — making them impractical for real-world, reference-free deployment. Modern LLM-as-judge techniques (e.g., G-Eval) are expensive and subjective.

This project proposes a **probability-based**, fully automated alternative: using an LLM's own internal confidence — measured via **conditional log-likelihood P(S|T)** — to evaluate and iteratively repair its own summaries, without any human-written references.

---

## 🧠 Core Idea

> *If a summary S is factually consistent with the source text T, the LLM should assign high conditional probability P(S|T) — i.e., low perplexity.*

We treat **low perplexity** as a proxy for **factual alignment**, and build an automated pipeline that:
1. Scores a generated summary via token-level log-likelihood
2. Flags high-perplexity (potentially hallucinated) summaries
3. Iteratively prompts the LLM to revise until consistency plateaus

---

## 🏗️ System Architecture

```
Source Document (T)
        │
        ▼
  LLM Summary Generation  ←─────────────────────────┐
        │                                            │
        ▼                                            │
  Log-Likelihood Scoring                             │
  P(S|T) via open-weight LLM                        │
        │                                            │
        ▼                                            │
  Weighted Perplexity                                │
  Calculation                                        │
  (key tokens up-weighted)                           │
        │                                            │
   Perplexity > threshold?                           │
        │                                            │
       YES ────────── Prompt LLM to revise ──────────┘
        │
       NO
        │
        ▼
  Final Refined Summary ✓
```

---

## ⚙️ Methodology

### Phase 1 — Baseline Likelihood Metric
- Compute **P(S|T)** using modern open-weight models: **Llama-3**, **Qwen**, **Mistral**
- Benchmark correlation against human-judgment datasets: **SummEval**, **Eval4NLP**
- Avoid outdated models (e.g., GPT-2) in favour of current frontier open-weight alternatives

### Phase 2 — Weighted Perplexity
Standard perplexity assigns equal weight to all tokens — including generic stop-words ("the", "and", "is") that carry no factual content. We address the **likelihood bias** problem by:
- Identifying **factual key tokens** (named entities, numbers, domain-specific terms)
- Computing a **weighted perplexity** that up-weights these tokens over generic stop-words
- This improves sensitivity to hallucinations versus existing approaches like **BARTScore**

### Phase 3 — Iterative Refinement Loop
```python
def iterative_refine(source, summary, model, threshold, max_iter=5):
    for i in range(max_iter):
        perplexity = compute_weighted_perplexity(summary, source, model)
        if perplexity <= threshold:
            break
        summary = prompt_llm_to_revise(source, summary, model)
    return summary
```
- Loop terminates when perplexity converges to a **stable optimal plateau**
- Measures **number of iterations** to convergence as a proxy for hallucination severity

### Phase 4 — Long-Context Optimization
- Standard perplexity breaks down on very long documents due to the **"Lost-in-the-Middle"** phenomenon
- Pipeline stress-tested on documents exceeding **100K tokens**
- Validates that perplexity-guided refinement can repair hallucinations in long-form contexts

---

## 🔑 Key Differentiators vs Prior Work

| Method | Reference-Free | Iterative Repair | Handles Key Tokens | Long-Context |
|---|---|---|---|---|
| ROUGE / BLEU | ❌ | ❌ | ❌ | ❌ |
| BARTScore | ✅ | ❌ | ❌ | ❌ |
| G-Eval (LLM-as-judge) | ✅ | ❌ | ✅ | ❌ |
| **Ours (Perplexity-Guided)** | ✅ | ✅ | ✅ | ✅ |

---

## 📊 Evaluation

- **Correlation benchmarks:** SummEval, Eval4NLP (human-judgment datasets)
- **Baseline comparisons:** BERTScore, G-Eval, BARTScore
- **Long-context benchmarks:** Documents > 100K tokens
- **Expected output:** Comprehensive evaluation table showing strong correlation with human judgment

---

## 🛠️ Tech Stack

| Component | Tools |
|---|---|
| **LLM Inference** | Hugging Face Transformers, vLLM |
| **Models** | Llama-3, Qwen, Mistral (open-weight) |
| **Log-likelihood Extraction** | PyTorch, token-level logits |
| **Datasets** | SummEval, Eval4NLP |
| **Pipeline Orchestration** | Python, custom prompt-chaining |
| **Environment** | Jupyter Notebook, Google Colab |

---

## 📁 Repository Structure

```
├── src/
│   ├── perplexity.py          # Weighted perplexity computation
│   ├── refinement_loop.py     # Iterative prompt-chaining pipeline
│   ├── key_token_extractor.py # Factual token identification
│   └── evaluate.py            # Benchmark correlation scripts
├── notebooks/
│   ├── baseline_experiments.ipynb
│   ├── weighted_perplexity_analysis.ipynb
│   └── long_context_stress_test.ipynb
├── data/
│   ├── summeval/
│   └── eval4nlp/
├── results/
│   └── evaluation_table.csv
├── paper/
│   └── acl2026_draft.pdf
└── README.md
```

---

## 📄 Paper

This work is being prepared for submission to the **ACL 2026 Student Research Workshop**, proposing a novel iterative probability-optimization approach to factual consistency in abstractive summarization.

> **Title:** *Perplexity-Guided Iterative Refinement for Hallucination Mitigation in LLM-Generated Summaries*
> **Venue Target:** ACL 2026 Student Research Workshop

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Princekaga/Iterative-Summarization-Refinement
cd Iterative-Summarization-Refinement

# Install dependencies
pip install -r requirements.txt

# Run the refinement pipeline
python src/refinement_loop.py \
  --model "meta-llama/Llama-3-8b" \
  --source "path/to/source_doc.txt" \
  --threshold 3.5 \
  --max_iter 5
```

---

## 📬 Contact

**Prince Agarwal**
B.Tech, IIT Kanpur
📧 kprince24@iitk.ac.in

---

*This project is part of the Bharat Nirman Labs research initiative at IIT Kanpur.*
