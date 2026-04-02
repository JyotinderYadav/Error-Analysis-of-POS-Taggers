"""
data_loader.py
Universal Dependencies CoNLL-U parser and dataset utilities.
Course Project: Error Analysis of POS Taggers
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    """Represents a single token in CoNLL-U format (10 fields)."""
    id: str        # Token ID (may be range for MWT or decimal for empty nodes)
    form: str      # Word form
    lemma: str     # Lemma
    upos: str      # Universal POS tag
    xpos: str      # Language-specific POS tag
    feats: str     # Morphological features
    head: str      # Head token index
    deprel: str    # Dependency relation
    deps: str      # Enhanced dependency graph
    misc: str      # Miscellaneous

    @property
    def is_regular(self) -> bool:
        """True if this is a regular token (not multiword or empty)."""
        return '-' not in str(self.id) and '.' not in str(self.id)


@dataclass
class Sentence:
    """Represents a full sentence with metadata and tokens."""
    sent_id: str
    text: str
    tokens: List[Token] = field(default_factory=list)

    @property
    def regular_tokens(self) -> List[Token]:
        """Return only regular tokens (no MWT or empty nodes)."""
        return [t for t in self.tokens if t.is_regular and t.upos != '_']

    @property
    def words(self) -> List[str]:
        return [t.form for t in self.regular_tokens]

    @property
    def upos_tags(self) -> List[str]:
        return [t.upos for t in self.regular_tokens]

    @property
    def lemmas(self) -> List[str]:
        return [t.lemma for t in self.regular_tokens]


# ──────────────────────────────────────────────────────────────────────────────
# CoNLL-U Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_conllu(filepath: str) -> List[Sentence]:
    """
    Parse a CoNLL-U file and return a list of Sentence objects.
    Handles multiword tokens and empty nodes gracefully.
    """
    sentences: List[Sentence] = []
    current_tokens: List[Token] = []
    sent_id = ""
    text = ""

    with open(filepath, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')

            if line.startswith('# sent_id'):
                sent_id = line.split('=', 1)[1].strip()

            elif line.startswith('# text'):
                text = line.split('=', 1)[1].strip()

            elif line == '':
                if current_tokens:
                    sentences.append(Sentence(
                        sent_id=sent_id,
                        text=text,
                        tokens=current_tokens
                    ))
                current_tokens = []
                sent_id = ""
                text = ""

            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) == 10:
                    current_tokens.append(Token(*parts))

    # Last sentence (if no trailing blank line)
    if current_tokens:
        sentences.append(Sentence(sent_id=sent_id, text=text, tokens=current_tokens))

    return sentences


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_ud_dataset(
    data_dir: str,
    language: str = 'en',
    treebank: str = 'ewt'
) -> Dict[str, List[Sentence]]:
    """
    Load UD train / dev / test splits from a local directory.

    Expected filenames (standard UD naming):
        en_ewt-ud-train.conllu
        en_ewt-ud-dev.conllu
        en_ewt-ud-test.conllu

    Returns a dict keyed by split name.
    """
    splits: Dict[str, List[Sentence]] = {}
    for split in ['train', 'dev', 'test']:
        fname = f"{language}_{treebank}-ud-{split}.conllu"
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            sents = parse_conllu(fpath)
            splits[split] = sents
            print(f"  Loaded {len(sents):>5} sentences  ({split})")
        else:
            print(f"  WARNING: {fpath} not found — skipping {split}")
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def flatten_sentences(
    sentences: List[Sentence]
) -> Tuple[List[str], List[str], List[int]]:
    """
    Flatten a list of Sentence objects into:
        words       — list of all word forms
        gold_tags   — corresponding UPOS tags
        boundaries  — cumulative end index of each sentence (for position analysis)
    """
    words: List[str] = []
    gold_tags: List[str] = []
    boundaries: List[int] = []

    for sent in sentences:
        words.extend(sent.words)
        gold_tags.extend(sent.upos_tags)
        boundaries.append(len(words))

    return words, gold_tags, boundaries


def build_vocab_stats(
    sentences: List[Sentence]
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    From a list of sentences, build:
        word_counts     — {lowercase_word: total_count}
        word_tag_counts — {lowercase_word: {upos_tag: count}}

    Used for OOV detection and ambiguity analysis.
    """
    word_counts: Counter = Counter()
    word_tag_counts: Dict[str, Counter] = defaultdict(Counter)

    for sent in sentences:
        for token in sent.regular_tokens:
            w = token.form.lower()
            word_counts[w] += 1
            word_tag_counts[w][token.upos] += 1

    return dict(word_counts), {k: dict(v) for k, v in word_tag_counts.items()}


def dataset_statistics(sentences: List[Sentence]) -> Dict:
    """Return descriptive statistics for a split."""
    all_words = [t.form for s in sentences for t in s.regular_tokens]
    all_tags  = [t.upos for s in sentences for t in s.regular_tokens]
    sent_lens = [len(s.regular_tokens) for s in sentences]

    tag_dist = Counter(all_tags)
    return {
        'n_sentences': len(sentences),
        'n_tokens': len(all_words),
        'vocab_size': len(set(w.lower() for w in all_words)),
        'avg_sent_len': sum(sent_lens) / len(sent_lens) if sent_lens else 0,
        'max_sent_len': max(sent_lens) if sent_lens else 0,
        'tag_distribution': dict(tag_dist.most_common()),
    }
