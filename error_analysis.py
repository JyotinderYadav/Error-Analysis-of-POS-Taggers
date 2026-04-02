"""
error_analysis.py
All error-analysis logic for the POS tagging project.

Classes
-------
ErrorAnalyzer       — per-tagger metrics, confusion, OOV, ambiguity, position
CrossTaggerAnalysis — ensemble, hard-cases, error taxonomy, agreement
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Single-Tagger Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class ErrorAnalyzer:
    """
    Comprehensive error analysis for a single POS tagger.

    Parameters
    ----------
    gold_tags   : ground-truth UPOS labels
    pred_tags   : tagger predictions (same length)
    words       : word forms (optional; required for OOV / ambiguity analysis)
    """

    def __init__(
        self,
        gold_tags: List[str],
        pred_tags: List[str],
        words: Optional[List[str]] = None
    ):
        assert len(gold_tags) == len(pred_tags), (
            f"Length mismatch: gold={len(gold_tags)}, pred={len(pred_tags)}"
        )
        self.gold  = gold_tags
        self.pred  = pred_tags
        self.words = words
        self._errors: Optional[List[Tuple[int, str, str]]] = None  # lazy cache

    # ── Basic Metrics ─────────────────────────────────────────────────────────

    @property
    def errors(self) -> List[Tuple[int, str, str]]:
        """List of (index, gold_tag, pred_tag) for every incorrect prediction."""
        if self._errors is None:
            self._errors = [
                (i, g, p)
                for i, (g, p) in enumerate(zip(self.gold, self.pred))
                if g != p
            ]
        return self._errors

    @property
    def accuracy(self) -> float:
        correct = sum(g == p for g, p in zip(self.gold, self.pred))
        return correct / len(self.gold)

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy

    # ── Confusion Matrix ──────────────────────────────────────────────────────

    def confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build and return (matrix, tag_list).
        Rows = gold (true), Columns = predicted.
        """
        tags = sorted(set(self.gold) | set(self.pred))
        t2i  = {t: i for i, t in enumerate(tags)}
        n    = len(tags)
        mat  = np.zeros((n, n), dtype=int)

        for g, p in zip(self.gold, self.pred):
            if g in t2i and p in t2i:
                mat[t2i[g]][t2i[p]] += 1

        return mat, tags

    # ── Per-Tag Metrics ───────────────────────────────────────────────────────

    def per_tag_metrics(self) -> Dict[str, Dict]:
        """
        Compute precision, recall, F1, support for every UPOS tag.
        Returns {tag: {precision, recall, f1, support, tp, fp, fn}}.
        """
        metrics: Dict[str, Dict] = {}
        for tag in sorted(set(self.gold)):
            tp = sum(1 for g, p in zip(self.gold, self.pred) if g == tag and p == tag)
            fp = sum(1 for g, p in zip(self.gold, self.pred) if g != tag and p == tag)
            fn = sum(1 for g, p in zip(self.gold, self.pred) if g == tag and p != tag)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0
            )
            support = sum(1 for g in self.gold if g == tag)

            metrics[tag] = dict(
                precision=precision, recall=recall, f1=f1,
                support=support, tp=tp, fp=fp, fn=fn
            )
        return metrics

    # ── Top Confusion Pairs ───────────────────────────────────────────────────

    def most_confused_pairs(
        self, top_k: int = 10
    ) -> List[Tuple[str, str, int]]:
        """Return top-k (gold, pred, count) pairs ranked by frequency."""
        counter = Counter(
            (g, p) for g, p in zip(self.gold, self.pred) if g != p
        )
        return [(g, p, c) for (g, p), c in counter.most_common(top_k)]

    # ── OOV Analysis ─────────────────────────────────────────────────────────

    def oov_analysis(self, train_vocab: Set[str]) -> Dict:
        """
        Split error rate into OOV vs in-vocabulary.
        OOV = word (lowercased) not seen in training vocabulary.
        """
        if self.words is None:
            return {}

        oov_err = oov_tot = iv_err = iv_tot = 0
        for word, g, p in zip(self.words, self.gold, self.pred):
            if word.lower() not in train_vocab:
                oov_tot += 1
                if g != p:
                    oov_err += 1
            else:
                iv_tot += 1
                if g != p:
                    iv_err += 1

        return {
            'oov_error_rate': oov_err / oov_tot if oov_tot else 0.0,
            'iv_error_rate':  iv_err  / iv_tot  if iv_tot  else 0.0,
            'oov_total': oov_tot,
            'iv_total':  iv_tot,
            'oov_errors': oov_err,
            'iv_errors':  iv_err,
        }

    # ── Frequency-Bucket Analysis ─────────────────────────────────────────────

    def frequency_bucket_analysis(
        self, vocab_counts: Dict[str, int], buckets: List[int] = None
    ) -> Dict[str, Dict]:
        """
        Bin words into frequency buckets and compute error rate per bucket.
        buckets: upper bounds, e.g. [1, 5, 20, 100, inf]
        """
        if self.words is None:
            return {}
        if buckets is None:
            buckets = [1, 5, 20, 100, float('inf')]

        bucket_labels = [
            '1', '2–5', '6–20', '21–100', '>100'
        ]
        result = {lbl: {'errors': 0, 'total': 0} for lbl in bucket_labels}

        for word, g, p in zip(self.words, self.gold, self.pred):
            count = vocab_counts.get(word.lower(), 0)
            for i, upper in enumerate(buckets):
                if count <= upper:
                    lbl = bucket_labels[i]
                    result[lbl]['total'] += 1
                    if g != p:
                        result[lbl]['errors'] += 1
                    break

        for lbl, vals in result.items():
            vals['error_rate'] = (
                vals['errors'] / vals['total'] if vals['total'] else 0.0
            )
        return result

    # ── Lexical Ambiguity Analysis ────────────────────────────────────────────

    def ambiguous_word_analysis(
        self, word_tag_counts: Dict[str, Dict[str, int]]
    ) -> Dict:
        """
        Compare error rates for ambiguous (≥2 POS tags in training) vs
        unambiguous words.
        """
        if self.words is None:
            return {}

        amb_err = amb_tot = unamb_err = unamb_tot = 0
        for word, g, p in zip(self.words, self.gold, self.pred):
            tag_dist = word_tag_counts.get(word.lower(), {})
            if len(tag_dist) > 1:
                amb_tot += 1
                if g != p:
                    amb_err += 1
            else:
                unamb_tot += 1
                if g != p:
                    unamb_err += 1

        return {
            'ambiguous_error_rate':   amb_err   / amb_tot   if amb_tot   else 0.0,
            'unambiguous_error_rate': unamb_err / unamb_tot if unamb_tot else 0.0,
            'ambiguous_total':   amb_tot,
            'unambiguous_total': unamb_tot,
        }

    # ── Sentence-Position Analysis ────────────────────────────────────────────

    def sentence_position_analysis(
        self, boundaries: List[int]
    ) -> Dict[str, float]:
        """
        Compute error rate by relative position in sentence.
        boundaries: cumulative token-end indices per sentence.
        """
        bin_errors = defaultdict(int)
        bin_totals = defaultdict(int)

        prev = 0
        for boundary in boundaries:
            sent_len = boundary - prev
            if sent_len == 0:
                prev = boundary
                continue
            for offset in range(sent_len):
                idx = prev + offset
                if idx >= len(self.gold):
                    break
                rel = offset / sent_len
                if rel < 0.25:
                    bucket = 'start (0–25%)'
                elif rel < 0.5:
                    bucket = 'early-mid (25–50%)'
                elif rel < 0.75:
                    bucket = 'late-mid (50–75%)'
                else:
                    bucket = 'end (75–100%)'
                bin_totals[bucket] += 1
                if self.gold[idx] != self.pred[idx]:
                    bin_errors[bucket] += 1
            prev = boundary

        return {
            bkt: bin_errors[bkt] / bin_totals[bkt]
            for bkt in bin_totals
        }

    # ── Summary Dict ─────────────────────────────────────────────────────────

    def full_report(
        self,
        train_vocab: Optional[Set[str]] = None,
        vocab_counts: Optional[Dict[str, int]] = None,
        word_tag_counts: Optional[Dict[str, Dict[str, int]]] = None,
        boundaries: Optional[List[int]] = None,
    ) -> Dict:
        """Run all analyses and return a combined result dict."""
        report: Dict = {
            'accuracy':          self.accuracy,
            'error_rate':        self.error_rate,
            'n_total':           len(self.gold),
            'n_errors':          len(self.errors),
            'per_tag_metrics':   self.per_tag_metrics(),
            'top_confused_pairs': self.most_confused_pairs(10),
        }
        if train_vocab is not None:
            report['oov_analysis'] = self.oov_analysis(train_vocab)
        if vocab_counts is not None:
            report['frequency_analysis'] = self.frequency_bucket_analysis(vocab_counts)
        if word_tag_counts is not None:
            report['ambiguity_analysis'] = self.ambiguous_word_analysis(word_tag_counts)
        if boundaries is not None:
            report['position_analysis'] = self.sentence_position_analysis(boundaries)
        return report


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Tagger Analysis (Innovation Component)
# ──────────────────────────────────────────────────────────────────────────────

class CrossTaggerAnalysis:
    """
    Analysis across multiple taggers simultaneously.

    Provides:
      - Pairwise tagger agreement
      - Hard cases (all taggers wrong)
      - Error category breakdown (1-wrong / majority-wrong / all-wrong)
      - Majority-vote ensemble
      - Error taxonomy (OOV / Ambiguity / Punctuation / Proper Noun / Other)
    """

    def __init__(
        self,
        gold: List[str],
        predictions: Dict[str, List[str]],
        words: Optional[List[str]] = None
    ):
        self.gold        = gold
        self.predictions = predictions
        self.words       = words
        self.names       = list(predictions.keys())

    # ── Pairwise Agreement ────────────────────────────────────────────────────

    def pairwise_agreement(self) -> Dict[str, float]:
        """
        Compute Cohen's kappa-style agreement for every tagger pair.
        Returns {'{n1}_vs_{n2}': agreement_score}.
        """
        result: Dict[str, float] = {}
        for i, n1 in enumerate(self.names):
            for j, n2 in enumerate(self.names):
                if i >= j:
                    continue
                p1, p2 = self.predictions[n1], self.predictions[n2]
                agree  = sum(a == b for a, b in zip(p1, p2)) / len(p1)
                result[f"{n1}_vs_{n2}"] = round(agree, 4)
        return result

    # ── Hard Cases ────────────────────────────────────────────────────────────

    def hard_cases(self) -> List[Dict]:
        """
        Tokens where every tagger predicts incorrectly.
        Returns list of {index, word, gold, predictions_dict}.
        """
        preds_list = [(n, self.predictions[n]) for n in self.names]
        hard = []
        for i, gold in enumerate(self.gold):
            if all(pred[i] != gold for _, pred in preds_list):
                entry: Dict = {
                    'index': i,
                    'gold':  gold,
                    'predictions': {n: pred[i] for n, pred in preds_list},
                }
                if self.words:
                    entry['word'] = self.words[i]
                hard.append(entry)
        return hard

    # ── Error Category Breakdown ──────────────────────────────────────────────

    def error_category_breakdown(self) -> Dict[str, int]:
        """
        Classify each token by how many taggers got it wrong.

        Categories
        ----------
        all_correct       : 0 taggers wrong
        only_one_wrong    : exactly 1 tagger wrong
        majority_wrong    : more than 1 but not all wrong
        all_wrong         : all taggers wrong
        """
        cats: Counter = Counter()
        n = len(self.names)

        for i, gold in enumerate(self.gold):
            n_wrong = sum(
                1 for name in self.names if self.predictions[name][i] != gold
            )
            if n_wrong == 0:
                cats['all_correct'] += 1
            elif n_wrong == 1:
                cats['only_one_wrong'] += 1
            elif n_wrong < n:
                cats['majority_wrong'] += 1
            else:
                cats['all_wrong'] += 1

        return dict(cats)

    # ── Ensemble ─────────────────────────────────────────────────────────────

    def ensemble_majority_vote(self) -> List[str]:
        """
        Simple majority-vote ensemble: the tag predicted by the plurality
        of taggers wins.  Tie-breaking: alphabetical order (deterministic).
        """
        ensemble: List[str] = []
        for i in range(len(self.gold)):
            votes = Counter(self.predictions[n][i] for n in self.names)
            ensemble.append(votes.most_common(1)[0][0])
        return ensemble

    # ── Error Taxonomy ────────────────────────────────────────────────────────

    def error_taxonomy(
        self,
        word_tag_counts: Dict[str, Dict[str, int]],
        train_vocab: Set[str],
    ) -> Dict[str, Dict[str, int]]:
        """
        Classify every error made by each tagger into one of:
            OOV             — word not in training vocabulary
            Lexical_Ambiguity — word has ≥2 POS in training data
            Proper_Noun     — gold tag is PROPN
            Punctuation     — gold tag is PUNCT
            Other           — none of the above

        Returns {tagger_name: {error_type: count}}.
        """
        if self.words is None:
            return {}

        taxonomy: Dict[str, Counter] = {n: Counter() for n in self.names}

        for name in self.names:
            preds = self.predictions[name]
            for i, (word, gold, pred) in enumerate(
                zip(self.words, self.gold, preds)
            ):
                if gold == pred:
                    continue
                w = word.lower()
                is_oov   = w not in train_vocab
                is_ambig = len(word_tag_counts.get(w, {})) > 1

                if is_oov:
                    taxonomy[name]['OOV'] += 1
                elif gold == 'PUNCT':
                    taxonomy[name]['Punctuation'] += 1
                elif gold == 'PROPN':
                    taxonomy[name]['Proper_Noun'] += 1
                elif is_ambig:
                    taxonomy[name]['Lexical_Ambiguity'] += 1
                else:
                    taxonomy[name]['Other'] += 1

        return {n: dict(c) for n, c in taxonomy.items()}

    # ── Most Ambiguous Words ──────────────────────────────────────────────────

    def most_error_prone_words(self, top_k: int = 20) -> List[Tuple]:
        """
        Find words that are most frequently mispredicted across all taggers.
        Returns [(word, gold_tag, total_errors, tagger_breakdown), ...].
        """
        if self.words is None:
            return []

        word_errors: Counter = Counter()
        for name in self.names:
            preds = self.predictions[name]
            for word, gold, pred in zip(self.words, self.gold, preds):
                if gold != pred:
                    word_errors[(word, gold)] += 1

        result = []
        for (word, gold), count in word_errors.most_common(top_k):
            breakdown = {
                n: self.predictions[n][i]
                for n in self.names
                for i, (w, g) in enumerate(zip(self.words, self.gold))
                if w == word and g == gold
            }
            result.append((word, gold, count, breakdown))
        return result
