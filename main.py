#!/usr/bin/env python3
"""
main.py — POS Tagger Error Analysis Pipeline
=============================================
Course Project: Error Analysis of POS Taggers
Dataset : Universal Dependencies — English EWT
          https://universaldependencies.org/

Usage
-----
    # First time: download UD data
    python main.py --download

    # Full run
    python main.py

    # Quick smoke-test (100 sentences)
    python main.py --max-sentences 100

    # Custom data directory
    python main.py --data-dir /path/to/ud/en_ewt

Pipeline
--------
  1. Load UD data (train / dev / test)
  2. Compute vocabulary statistics from training split
  3. Load all four POS taggers
  4. Run each tagger on the test split
  5. Individual error analysis per tagger
  6. Cross-tagger analysis (ensemble, hard cases, taxonomy)
  7. Generate all plots → results/
  8. Save JSON summary → results/analysis_results.json
  9. Print summary table
"""

import os
import sys
import json
import argparse
from collections import Counter
from typing import List, Dict, Optional

# ── Project modules ───────────────────────────────────────────────────────────
from data_loader import (
    load_ud_dataset,
    flatten_sentences,
    build_vocab_stats,
    dataset_statistics,
)
from taggers import get_available_taggers
from error_analysis import ErrorAnalyzer, CrossTaggerAnalysis
from visualizations import (
    plot_overall_accuracy,
    plot_per_tag_f1,
    plot_confusion_matrix,
    plot_top_confusion_pairs,
    plot_oov_analysis,
    plot_frequency_bucket,
    plot_ambiguity_analysis,
    plot_error_taxonomy,
    plot_error_category_breakdown,
    plot_ensemble_comparison,
    plot_position_analysis,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def download_ud_ewt(data_dir: str):
    """Download UD English EWT treebank from the official GitHub mirror."""
    os.makedirs(data_dir, exist_ok=True)
    base = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
    for split in ['train', 'dev', 'test']:
        fname = f'en_ewt-ud-{split}.conllu'
        dest  = os.path.join(data_dir, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname} ...", end=' ', flush=True)
        ret = os.system(f'curl -s -L -o "{dest}" "{base}/{fname}"')
        if ret == 0:
            print("OK")
        else:
            print(f"FAILED (exit code {ret})")


def tag_sentences_with(tagger, sentences) -> List[str]:
    """
    Run `tagger` on every sentence in `sentences` and flatten predictions
    into a single list aligned with the token order from flatten_sentences().
    """
    pred_tags: List[str] = []
    for sent in sentences:
        words = sent.words
        if not words:
            continue
        try:
            tags = tagger.tag_sentence(words)
            # Ensure exactly len(words) tags
            if len(tags) < len(words):
                tags += ['X'] * (len(words) - len(tags))
            tags = tags[:len(words)]
        except Exception as exc:
            print(f"    Warning — tagger error on sentence '{sent.sent_id}': {exc}")
            tags = ['X'] * len(words)
        pred_tags.extend(tags)
    return pred_tags


def serialise(obj):
    """Recursively convert non-JSON-serialisable objects to str / basic types."""
    if isinstance(obj, dict):
        return {str(k): serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialise(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


def print_banner(text: str):
    line = '─' * 60
    print(f"\n{line}\n  {text}\n{line}")


def print_summary_table(results: Dict, ensemble_acc: Optional[float]):
    print_banner("RESULTS SUMMARY")
    print(f"{'Tagger':<30} {'Accuracy':>10}  {'Error Rate':>12}")
    print('─' * 56)
    for name, data in results['taggers'].items():
        acc = data['accuracy']
        print(f"{name:<30} {acc*100:>9.2f}%  {(1-acc)*100:>11.2f}%")
    if ensemble_acc is not None:
        print(f"{'Ensemble (Majority Vote)':<30} {ensemble_acc*100:>9.2f}%  "
              f"{(1-ensemble_acc)*100:>11.2f}%")
    print('─' * 56)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='POS Tagger Error Analysis — Universal Dependencies'
    )
    parser.add_argument('--data-dir',       default='data/en_ewt-ud',
                        help='Directory containing en_ewt-ud-*.conllu files')
    parser.add_argument('--output-dir',     default='results',
                        help='Directory for plots and JSON output')
    parser.add_argument('--max-sentences',  type=int, default=None,
                        help='Cap on test sentences (useful for quick tests)')
    parser.add_argument('--download',       action='store_true',
                        help='Download UD English EWT data before running')
    parser.add_argument('--skip-hf',        action='store_true',
                        help='Skip HuggingFace tagger (saves time / memory)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 0: Download ────────────────────────────────────────────────────
    if args.download:
        print_banner("Downloading UD English EWT")
        download_ud_ewt(args.data_dir)

    # ── Step 1: Load Data ───────────────────────────────────────────────────
    print_banner("Loading Universal Dependencies (English EWT)")
    splits = load_ud_dataset(args.data_dir)

    if not splits:
        print(
            "\nERROR: No data found. "
            f"Place *.conllu files in '{args.data_dir}' or run with --download"
        )
        sys.exit(1)

    train_sents = splits.get('train', [])
    test_sents  = splits.get('test',  splits.get('dev', []))

    if args.max_sentences:
        test_sents = test_sents[:args.max_sentences]

    print(f"\n  Train: {len(train_sents):>5} sentences")
    print(f"  Test : {len(test_sents):>5} sentences")

    # Dataset statistics
    train_stats = dataset_statistics(train_sents)
    test_stats  = dataset_statistics(test_sents)
    print(f"\n  Train tokens : {train_stats['n_tokens']:,} | "
          f"Vocab: {train_stats['vocab_size']:,}")
    print(f"  Test  tokens : {test_stats['n_tokens']:,}")

    # Build vocabulary from training data
    vocab_counts, word_tag_counts = build_vocab_stats(train_sents)
    train_vocab = set(vocab_counts.keys())

    # Flatten test set
    test_words, gold_tags, boundaries = flatten_sentences(test_sents)
    print(f"\n  Test tokens (flat): {len(test_words):,}")

    # ── Step 2: Load Taggers ────────────────────────────────────────────────
    print_banner("Loading POS Taggers")
    taggers = get_available_taggers()

    if args.skip_hf and 'huggingface' in taggers:
        del taggers['huggingface']
        print("  (HuggingFace tagger skipped via --skip-hf)")

    if not taggers:
        print("ERROR: No taggers could be loaded. Install dependencies.")
        sys.exit(1)

    # ── Step 3: Run Taggers ─────────────────────────────────────────────────
    print_banner("Running Taggers on Test Set")
    all_predictions: Dict[str, List[str]] = {}

    for key, tagger in taggers.items():
        print(f"\n  → {tagger.name}")
        preds = tag_sentences_with(tagger, test_sents)
        # Trim to gold length for safety
        n = min(len(preds), len(gold_tags))
        all_predictions[tagger.name] = preds[:n]

    # Align everything to shortest
    n_align = min(len(gold_tags),
                  min(len(p) for p in all_predictions.values()))
    gold_aligned  = gold_tags[:n_align]
    words_aligned = test_words[:n_align]
    preds_aligned = {name: p[:n_align] for name, p in all_predictions.items()}

    print(f"\n  Aligned tokens: {n_align:,}")

    # ── Step 4: Individual Error Analysis ───────────────────────────────────
    print_banner("Individual Error Analysis")
    results: Dict = {
        'meta': {
            'dataset': 'Universal Dependencies — English EWT',
            'train_sentences': len(train_sents),
            'test_sentences':  len(test_sents),
            'test_tokens':     n_align,
            'taggers':         list(preds_aligned.keys()),
        },
        'taggers': {},
    }

    per_tag_for_plot: Dict[str, Dict] = {}

    for name, preds in preds_aligned.items():
        print(f"\n  Analyzing: {name}")
        analyzer = ErrorAnalyzer(gold_aligned, preds, words_aligned)
        report   = analyzer.full_report(
            train_vocab    = train_vocab,
            vocab_counts   = vocab_counts,
            word_tag_counts= word_tag_counts,
            boundaries     = boundaries[:len(test_sents)],
        )
        results['taggers'][name] = report
        per_tag_for_plot[name]   = report['per_tag_metrics']

        print(f"    Accuracy  : {report['accuracy']*100:.2f}%")
        print(f"    Errors    : {report['n_errors']:,} / {n_align:,}")

        # Confusion matrix
        mat, mat_tags = analyzer.confusion_matrix()
        plot_confusion_matrix(mat, mat_tags, name, args.output_dir)
        plot_top_confusion_pairs(
            report['top_confused_pairs'], name, args.output_dir
        )

    # Shared plots
    accs = {n: results['taggers'][n]['accuracy'] for n in preds_aligned}
    plot_per_tag_f1(per_tag_for_plot, args.output_dir)

    oov_data = {n: results['taggers'][n].get('oov_analysis', {}) for n in accs}
    plot_oov_analysis(oov_data, args.output_dir)

    freq_data = {n: results['taggers'][n].get('frequency_analysis', {}) for n in accs}
    plot_frequency_bucket(freq_data, args.output_dir)

    amb_data = {n: results['taggers'][n].get('ambiguity_analysis', {}) for n in accs}
    plot_ambiguity_analysis(amb_data, args.output_dir)

    pos_data = {n: results['taggers'][n].get('position_analysis', {}) for n in accs}
    plot_position_analysis(pos_data, args.output_dir)

    # ── Step 5: Cross-Tagger Analysis ──────────────────────────────────────
    print_banner("Cross-Tagger Analysis (Innovation)")
    cross = CrossTaggerAnalysis(gold_aligned, preds_aligned, words_aligned)

    # Ensemble
    ensemble_preds  = cross.ensemble_majority_vote()
    ensemble_anal   = ErrorAnalyzer(gold_aligned, ensemble_preds, words_aligned)
    ensemble_acc    = ensemble_anal.accuracy
    print(f"  Ensemble accuracy : {ensemble_acc*100:.2f}%")

    # Hard cases
    hard = cross.hard_cases()
    print(f"  Hard cases (all wrong): {len(hard):,}")

    # Error category breakdown
    breakdown = cross.error_category_breakdown()
    print(f"  Category breakdown: {breakdown}")

    # Pairwise agreement
    agreement = cross.pairwise_agreement()

    # Error taxonomy
    taxonomy = cross.error_taxonomy(word_tag_counts, train_vocab)

    results['cross_tagger'] = {
        'ensemble_accuracy':       ensemble_acc,
        'ensemble_error_rate':     1 - ensemble_acc,
        'n_hard_cases':            len(hard),
        'hard_case_sample':        hard[:15],
        'error_category_breakdown':breakdown,
        'pairwise_agreement':      agreement,
        'error_taxonomy':          {n: dict(t) for n, t in taxonomy.items()},
    }

    # Cross-tagger plots
    plot_overall_accuracy(accs, args.output_dir, ensemble_acc=ensemble_acc)
    plot_ensemble_comparison(accs, ensemble_acc, args.output_dir)
    plot_error_category_breakdown(breakdown, args.output_dir)
    plot_error_taxonomy(taxonomy, args.output_dir)

    # ── Step 6: Save Results ────────────────────────────────────────────────
    print_banner("Saving Results")
    out_json = os.path.join(args.output_dir, 'analysis_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(serialise(results), f, indent=2)
    print(f"  JSON results: {out_json}")

    # ── Step 7: Summary ─────────────────────────────────────────────────────
    print_summary_table(results, ensemble_acc)

    print(f"\n  Plots saved to: {os.path.abspath(args.output_dir)}/")
    print("\n  Done ✓\n")


if __name__ == '__main__':
    main()
