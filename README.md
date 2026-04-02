# Error Analysis of POS Taggers
### NLP Course Project | Universal Dependencies (English EWT)

---

## Overview

This project systematically compares four POS tagging systems on the
Universal Dependencies English EWT treebank, categorising and visualising
their errors across multiple dimensions.

| Component | Weight | What's covered |
|-----------|--------|---------------|
| Survey (Literature Review) | 10 % | `report/survey_report.docx` |
| Implementation | 20 % | `main.py`, `taggers.py`, `data_loader.py` |
| Analysis | 10 % | `error_analysis.py`, `results/` |
| Innovation | 10 % | Ensemble tagger, error taxonomy, cross-tagger hard-case analysis |

---

## Taggers Compared

| # | Tagger | Architecture | Tag scheme |
|---|--------|-------------|------------|
| 1 | **NLTK** Averaged Perceptron | Statistical | Penn Treebank → UPOS |
| 2 | **spaCy** `en_core_web_sm` | CNN neural | UPOS (direct) |
| 3 | **Stanza** | BiLSTM + CRF | UPOS (trained on UD) |
| 4 | **BERT** (HuggingFace) | Transformer | UPOS (fine-tuned on UD) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download the dataset

```bash
python main.py --download
```

This fetches the three CoNLL-U splits of UD English EWT into `data/en_ewt-ud/`.
Alternatively, download manually from https://universaldependencies.org/ and
place the files there.

### 3. Run the full pipeline

```bash
python main.py
```

### 4. Quick test (100 sentences)

```bash
python main.py --max-sentences 100
```

### 5. Skip HuggingFace (faster, saves memory)

```bash
python main.py --skip-hf
```

---

## Output

All outputs are saved to `results/` :

```
results/
├── analysis_results.json          ← complete metrics (machine-readable)
├── overall_accuracy.png           ← accuracy bar chart
├── per_tag_f1_comparison.png      ← per-UPOS F1 across all taggers
├── confusion_<tagger>.png         ← confusion matrices (one per tagger)
├── top_errors_<tagger>.png        ← top confusion pairs per tagger
├── oov_analysis.png               ← OOV vs in-vocabulary error rates
├── frequency_bucket_errors.png    ← error rate by word frequency
├── ambiguity_analysis.png         ← ambiguous vs unambiguous words
├── error_taxonomy.png             ← pie charts: OOV/Ambig/PROPN/PUNCT/Other
├── error_category_breakdown.png   ← how many taggers got each token wrong
├── ensemble_comparison.png        ← individual vs majority-vote ensemble
└── position_error_rates.png       ← error rate by position in sentence
```

---

## Project Structure

```
pos_error_analysis/
├── main.py            ← Pipeline orchestrator (run this)
├── data_loader.py     ← CoNLL-U parser + UD dataset utilities
├── taggers.py         ← Four tagger wrappers (common interface)
├── error_analysis.py  ← ErrorAnalyzer + CrossTaggerAnalysis classes
├── visualizations.py  ← All Matplotlib plotting functions
├── requirements.txt
└── README.md
```

---

## Analysis Dimensions

The project analyses errors along six dimensions:

1. **Overall accuracy and error rate** — headline numbers per tagger
2. **Per-tag precision / recall / F1** — which UPOS categories are hardest?
3. **Confusion matrices** — which tags get confused with which?
4. **OOV vs In-Vocabulary** — how much does unseen vocabulary hurt each tagger?
5. **Word-frequency buckets** — do rare words cause more errors?
6. **Lexical ambiguity** — are ambiguous words harder across all taggers?
7. **Sentence position** — do errors cluster at sentence boundaries?

### Innovation Components

- **Majority-vote ensemble** — combines all four taggers; compares accuracy gain
- **Hard cases** — tokens where *all* taggers fail → intrinsically hard examples
- **Error category breakdown** — classifies each token: 0/1/majority/all wrong
- **Error taxonomy** — tags every error as OOV, Lexical Ambiguity, Proper Noun,
  Punctuation, or Other; per-tagger breakdown
- **Cross-tagger pairwise agreement** — measures how much taggers agree

---

## Dataset

Universal Dependencies English EWT  
<https://universaldependencies.org/>  
License: CC BY-SA 4.0  

Splits (approximate):
- Train: ~12 500 sentences / 204 000 tokens
- Dev:   ~2 000 sentences / 25 000 tokens
- Test:  ~2 100 sentences / 25 000 tokens

---

## References

See `report/survey_report.docx` for full bibliography.  
Key papers:
- Nivre et al. (2016). Universal Dependencies v1.
- Akbik et al. (2018). Contextual String Embeddings for Sequence Labeling.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
- Qi et al. (2020). Stanza: A Python NLP Library for Many Human Languages.
