"""
taggers.py
Unified wrapper classes for four POS taggers.

Taggers implemented
-------------------
1. NLTKTagger        — Averaged Perceptron (statistical, PTB → UPOS)
2. SpacyTagger       — CNN-based neural (spaCy en_core_web_sm)
3. StanzaTagger      — BiLSTM + CRF neural (Stanford Stanza)
4. HuggingFaceTagger — BERT fine-tuned on UD POS (Transformers)

All taggers expose the same interface:
    tagger.tag_sentence(words: List[str]) -> List[str]   # returns UPOS tags
    tagger.tag_sentences(sentences)       -> List[List[str]]
"""

import abc
import os
import json
from typing import List, Dict
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

UPOS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
    'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
]

# Penn Treebank → Universal POS mapping (standard mapping from UD guidelines)
PTB_TO_UPOS: Dict[str, str] = {
    'CC': 'CCONJ',   'CD': 'NUM',    'DT': 'DET',    'EX': 'PRON',
    'FW': 'X',       'IN': 'ADP',    'JJ': 'ADJ',    'JJR': 'ADJ',
    'JJS': 'ADJ',    'LS': 'X',      'MD': 'AUX',    'NN': 'NOUN',
    'NNS': 'NOUN',   'NNP': 'PROPN', 'NNPS': 'PROPN','PDT': 'DET',
    'POS': 'PART',   'PRP': 'PRON',  'PRP$': 'PRON', 'RB': 'ADV',
    'RBR': 'ADV',    'RBS': 'ADV',   'RP': 'ADP',    'SYM': 'SYM',
    'TO': 'PART',    'UH': 'INTJ',   'VB': 'VERB',   'VBD': 'VERB',
    'VBG': 'VERB',   'VBN': 'VERB',  'VBP': 'VERB',  'VBZ': 'VERB',
    'WDT': 'DET',    'WP': 'PRON',   'WP$': 'PRON',  'WRB': 'ADV',
    '#': 'SYM',      '$': 'SYM',     '.': 'PUNCT',   ',': 'PUNCT',
    ':': 'PUNCT',    '``': 'PUNCT',  "''": 'PUNCT',  '-LRB-': 'PUNCT',
    '-RRB-': 'PUNCT','HYPH': 'PUNCT','AFX': 'ADJ',   'ADD': 'X',
    'GW': 'X',       'NFP': 'PUNCT', 'XX': 'X',      'NN|SYM': 'SYM',
    'NNPS|SYM': 'SYM','JJ|VBG': 'ADJ','VBG|JJ': 'VERB','-': 'PUNCT',
    'PRP|VBP': 'PRON','RB|VBG': 'ADV',
}


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────────────────────────────────────

class BaseTagger(abc.ABC):
    """All taggers must implement this interface."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable tagger name."""

    @abc.abstractmethod
    def tag_sentence(self, words: List[str]) -> List[str]:
        """
        Tag a single pre-tokenized sentence.
        Returns a list of UPOS tags of the same length as `words`.
        """

    def tag_sentences(self, sentences: List[List[str]]) -> List[List[str]]:
        """Tag multiple sentences (sequential by default)."""
        return [self.tag_sentence(words) for words in sentences]

    def _ensure_length(self, tags: List[str], target_len: int) -> List[str]:
        """Pad or truncate tag list to match target length."""
        if len(tags) < target_len:
            tags = tags + ['X'] * (target_len - len(tags))
        return tags[:target_len]


# ──────────────────────────────────────────────────────────────────────────────
# 1. NLTK Averaged Perceptron Tagger
# ──────────────────────────────────────────────────────────────────────────────

class NLTKTagger(BaseTagger):
    """
    NLTK's Averaged Perceptron Tagger.
    Uses Penn Treebank tagset → mapped to UPOS.
    Strengths : fast, no GPU required, good baseline.
    Weaknesses: PTB→UPOS mapping introduces noise; rule-based origins.
    """

    def __init__(self):
        import nltk
        for resource in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        from nltk import pos_tag as _tag
        self._tag = _tag

    @property
    def name(self) -> str:
        return "NLTK (AvgPerceptron)"

    def tag_sentence(self, words: List[str]) -> List[str]:
        if not words:
            return []
        tagged = self._tag(words)
        return [PTB_TO_UPOS.get(tag, 'X') for _, tag in tagged]


# ──────────────────────────────────────────────────────────────────────────────
# 2. spaCy CNN Tagger
# ──────────────────────────────────────────────────────────────────────────────

class SpacyTagger(BaseTagger):
    """
    spaCy POS tagger using a CNN architecture (en_core_web_sm).
    Already predicts UPOS directly — no mapping required.
    Strengths : fast neural tagger, good accuracy.
    Weaknesses: smaller model; less context than transformers.
    """

    def __init__(self, model: str = 'en_core_web_sm'):
        import spacy
        try:
            self._nlp = spacy.load(model, disable=['parser', 'ner', 'lemmatizer'])
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'spacy', 'download', model], check=True
            )
            self._nlp = spacy.load(model, disable=['parser', 'ner', 'lemmatizer'])
        self._model = model

    @property
    def name(self) -> str:
        return f"spaCy ({self._model})"

    def tag_sentence(self, words: List[str]) -> List[str]:
        if not words:
            return []
        from spacy.tokens import Doc
        doc = Doc(self._nlp.vocab, words=words)
        for _, proc in self._nlp.pipeline:
            if hasattr(proc, '__call__'):
                try:
                    proc(doc)
                except Exception:
                    pass
        tags = []
        for token in doc:
            pos = token.pos_
            if not pos or pos == 'NIL' or pos == '':
                pos = PTB_TO_UPOS.get(token.tag_, 'X')
            if pos not in UPOS_TAGS:
                pos = 'X'
            tags.append(pos)
        return self._ensure_length(tags, len(words))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Stanford Stanza (BiLSTM + CRF)
# ──────────────────────────────────────────────────────────────────────────────

class StanzaTagger(BaseTagger):
    """
    Stanford Stanza POS tagger — BiLSTM + CRF trained on UD.
    Directly predicts UPOS from Universal Dependencies training data.
    Strengths : trained on UD → minimal tag-mapping errors; strong accuracy.
    Weaknesses: slower than spaCy; requires Java-free Python runtime.
    """

    def __init__(self, lang: str = 'en'):
        import stanza
        stanza.download(lang, processors='tokenize,pos')
        self._nlp = stanza.Pipeline(
            lang,
            processors='tokenize,pos',
            tokenize_pretokenized=True,
            verbose=False,
            use_gpu=False
        )
        self._lang = lang

    @property
    def name(self) -> str:
        return "Stanza (BiLSTM+CRF)"

    def tag_sentence(self, words: List[str]) -> List[str]:
        if not words:
            return []
        doc = self._nlp([words])
        tags = []
        for sentence in doc.sentences:
            for word in sentence.words:
                tags.append(word.upos if word.upos else 'X')
        return self._ensure_length(tags, len(words))


# ──────────────────────────────────────────────────────────────────────────────
# 4. HuggingFace BERT-based Tagger
# ──────────────────────────────────────────────────────────────────────────────

class HuggingFaceTagger(BaseTagger):
    """
    BERT fine-tuned for UD POS tagging via HuggingFace Transformers.
    Model: vblagoje/bert-english-uncased-finetuned-pos
    Predicts UPOS tags using subword tokenization + alignment.

    Strengths : contextual representations; handles ambiguity well.
    Weaknesses: slower; requires GPU for speed; OOV handled via subwords.
    """

    MODEL_NAME = 'vblagoje/bert-english-uncased-finetuned-pos'

    def __init__(self, model_name: str = None):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch

        self._model_name = model_name or self.MODEL_NAME
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(self._model_name)
        self._model.eval()
        self._id2label = self._model.config.id2label
        self._torch = torch

    @property
    def name(self) -> str:
        return "BERT (HuggingFace)"

    def tag_sentence(self, words: List[str]) -> List[str]:
        if not words:
            return []

        # Tokenize with word_ids for alignment
        encoding = self._tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        with self._torch.no_grad():
            logits = self._model(**encoding).logits

        predictions = logits.argmax(dim=-1).squeeze().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]

        word_ids = encoding.word_ids(batch_index=0)

        # Aggregate subword predictions → word predictions (first subtoken rule)
        word_preds: Dict[int, str] = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_preds:
                raw_tag = self._id2label.get(predictions[token_idx], 'X')
                # Strip BIO prefix if present
                if '-' in raw_tag:
                    raw_tag = raw_tag.split('-', 1)[1]
                # Map PTB → UPOS if needed
                upos = PTB_TO_UPOS.get(raw_tag, raw_tag)
                if upos not in UPOS_TAGS:
                    upos = 'X'
                word_preds[word_idx] = upos

        tags = [word_preds.get(i, 'X') for i in range(len(words))]
        return self._ensure_length(tags, len(words))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Grok xAI Zero-Shot Tagger
# ──────────────────────────────────────────────────────────────────────────────

class GrokTagger(BaseTagger):
    """
    xAI Grok zero-shot POS tagger.
    Prompts the LLM with words and requests UPOS JSON array output.
    """
    def __init__(self, model_name: str = 'grok-beta'):
        self.api_key = os.getenv("GROK_API_KEY")
        self._model_name = model_name
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        if not self.api_key:
            raise ValueError("GROK_API_KEY environment variable not set")

    @property
    def name(self) -> str:
        return f"Grok ({self._model_name})"

    def tag_sentence(self, words: List[str]) -> List[str]:
        if not words:
            return []
            
        try:
            import requests
        except ImportError:
            print("    [Grok API Error] Please install 'requests' library (pip install requests).")
            return ['X'] * len(words)
        
        prompt = f"You are a UPOS tagger. Given a list of {len(words)} English words, output a JSON array of strings containing exactly {len(words)} UPOS tags ordered correspondingly. Do not explain, just return the JSON array.\nWords: {json.dumps(words)}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": "You are a specialized NLP tagger returning raw JSON arrays. Adhere perfectly to UPOS tags."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
        try:
            resp = requests.post(self.endpoint, headers=headers, json=data, timeout=15)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'].strip()
            
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                content = match.group(0)
            
            tags = json.loads(content)
            if not isinstance(tags, list):
                tags = ['X'] * len(words)
                
            tags = [t if t in UPOS_TAGS else PTB_TO_UPOS.get(t, 'X') for t in tags]
            return self._ensure_length(tags, len(words))
            
        except Exception as e:
            print(f"    [Grok API Error] {e}")
            if 'content' in locals():
                print(f"    [Grok Content] {content[:200]}")
            return ['X'] * len(words)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_available_taggers() -> Dict[str, BaseTagger]:
    """
    Try to initialise all taggers.
    Skips any that fail to load (prints reason).
    Returns a dict of {key: tagger_instance}.
    """
    candidates = [
        ('nltk',        NLTKTagger),
        ('spacy',       SpacyTagger),
        ('stanza',      StanzaTagger),
        ('huggingface', HuggingFaceTagger),
        ('grok',        GrokTagger),
    ]

    taggers: Dict[str, BaseTagger] = {}
    for key, cls in candidates:
        try:
            print(f"  Loading {cls.__name__}... ", end='', flush=True)
            taggers[key] = cls()
            print(f"OK  →  {taggers[key].name}")
        except Exception as exc:
            print(f"FAILED ({exc})")

    return taggers
