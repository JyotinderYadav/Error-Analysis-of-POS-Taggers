"""
visualizations.py
All Matplotlib / Seaborn plotting functions for the POS error analysis project.
Each function saves a PNG to output_dir and returns its path.
"""

import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = ['#1976D2', '#388E3C', '#D32F2F', '#7B1FA2', '#F57C00', '#0097A7']
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F9FA',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})


def _save(fig: plt.Figure, path: str) -> str:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1. Overall Accuracy Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_overall_accuracy(
    accuracies: Dict[str, float],
    output_dir: str,
    ensemble_acc: Optional[float] = None
) -> str:
    names  = list(accuracies.keys())
    values = [v * 100 for v in accuracies.values()]
    colors = PALETTE[:len(names)]

    if ensemble_acc is not None:
        names  += ['Ensemble\n(Majority Vote)']
        values += [ensemble_acc * 100]
        colors += ['#FFB300']

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 6))
    bars = ax.bar(names, values, color=colors, alpha=0.88,
                  edgecolor='white', linewidth=1.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f'{val:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    ax.set_ylim(max(0, min(values) - 5), 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall POS Tagging Accuracy — Universal Dependencies (EWT)')
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.axhline(95, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(names) - 0.5, 95.3, '95 %', color='gray', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F9FA')

    return _save(fig, os.path.join(output_dir, 'overall_accuracy.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-Tag F1 Grouped Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_tag_f1(
    metrics_by_tagger: Dict[str, Dict],
    output_dir: str
) -> str:
    all_tags     = sorted({tag for m in metrics_by_tagger.values() for tag in m})
    tagger_names = list(metrics_by_tagger.keys())
    n_taggers    = len(tagger_names)

    x     = np.arange(len(all_tags))
    width = 0.8 / n_taggers

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, name in enumerate(tagger_names):
        f1_scores = [metrics_by_tagger[name].get(tag, {}).get('f1', 0)
                     for tag in all_tags]
        offset = (i - n_taggers / 2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width * 0.9,
               label=name, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xlabel('Universal POS Tag')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Tag F1 Score Comparison Across POS Taggers')
    ax.set_xticks(x)
    ax.set_xticklabels(all_tags, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    return _save(fig, os.path.join(output_dir, 'per_tag_f1_comparison.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    matrix: np.ndarray,
    tags: List[str],
    tagger_name: str,
    output_dir: str
) -> str:
    row_sums    = matrix.sum(axis=1, keepdims=True)
    normalized  = np.where(row_sums > 0, matrix / row_sums, 0)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    n = len(tags)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tags, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tags, fontsize=9)
    ax.set_xlabel('Predicted Tag')
    ax.set_ylabel('Gold (True) Tag')
    ax.set_title(f'Confusion Matrix — {tagger_name}\n(row-normalised: shows recall per tag)')

    for i in range(n):
        for j in range(n):
            val = normalized[i, j]
            if val > 0.02:
                color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color,
                        fontweight='bold' if i == j else 'normal')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    safe_name = tagger_name.replace(' ', '_').replace('(', '').replace(')', '')
    return _save(fig, os.path.join(output_dir, f'confusion_{safe_name}.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Top Confusion Pairs (Horizontal Bar)
# ──────────────────────────────────────────────────────────────────────────────

def plot_top_confusion_pairs(
    pairs: List[Tuple[str, str, int]],
    tagger_name: str,
    output_dir: str
) -> Optional[str]:
    if not pairs:
        return None

    labels = [f"{g} → {p}" for g, p, _ in pairs]
    counts = [c for _, _, c in pairs]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(labels)), counts, color='#1565C0', alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Count of Errors')
    ax.set_title(f'Top Confusion Pairs — {tagger_name}')
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    safe_name = tagger_name.replace(' ', '_').replace('(', '').replace(')', '')
    return _save(fig, os.path.join(output_dir, f'top_errors_{safe_name}.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 5. OOV vs In-Vocabulary Error Rate
# ──────────────────────────────────────────────────────────────────────────────

def plot_oov_analysis(
    oov_data: Dict[str, Dict],
    output_dir: str
) -> str:
    names    = list(oov_data.keys())
    oov_rates = [oov_data[n].get('oov_error_rate', 0) * 100 for n in names]
    iv_rates  = [oov_data[n].get('iv_error_rate',  0) * 100 for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, oov_rates, width, label='OOV Words',
           color='#D32F2F', alpha=0.85)
    ax.bar(x + width/2, iv_rates,  width, label='In-Vocabulary',
           color='#1976D2', alpha=0.85)

    ax.set_ylabel('Error Rate (%)')
    ax.set_title('OOV vs In-Vocabulary Word Error Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return _save(fig, os.path.join(output_dir, 'oov_analysis.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Frequency-Bucket Error Rates
# ──────────────────────────────────────────────────────────────────────────────

def plot_frequency_bucket(
    freq_data: Dict[str, Dict[str, Dict]],
    output_dir: str
) -> str:
    """freq_data: {tagger_name: {bucket_label: {error_rate, total}}}"""
    tagger_names = list(freq_data.keys())
    if not tagger_names:
        return ''

    buckets = list(next(iter(freq_data.values())).keys())
    x       = np.arange(len(buckets))
    width   = 0.8 / len(tagger_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(tagger_names):
        rates  = [freq_data[name].get(b, {}).get('error_rate', 0) * 100
                  for b in buckets]
        offset = (i - len(tagger_names) / 2 + 0.5) * width
        ax.bar(x + offset, rates, width * 0.9,
               label=name, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xlabel('Word Frequency in Training Data (occurrences)')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate by Word Frequency Bucket')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    return _save(fig, os.path.join(output_dir, 'frequency_bucket_errors.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 7. Ambiguity Analysis (stacked bar)
# ──────────────────────────────────────────────────────────────────────────────

def plot_ambiguity_analysis(
    amb_data: Dict[str, Dict],
    output_dir: str
) -> str:
    names     = list(amb_data.keys())
    amb_rates = [amb_data[n].get('ambiguous_error_rate',   0) * 100 for n in names]
    un_rates  = [amb_data[n].get('unambiguous_error_rate', 0) * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, amb_rates, w, label='Ambiguous words', color='#E64A19', alpha=0.85)
    ax.bar(x + w/2, un_rates,  w, label='Unambiguous words', color='#1976D2', alpha=0.85)

    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Lexically Ambiguous vs Unambiguous Word Error Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    return _save(fig, os.path.join(output_dir, 'ambiguity_analysis.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 8. Error Taxonomy Pie Charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_error_taxonomy(
    taxonomy: Dict[str, Dict[str, int]],
    output_dir: str
) -> str:
    """taxonomy: {tagger_name: {error_type: count}}"""
    names = list(taxonomy.keys())
    if not names:
        return ''

    ncols = min(len(names), 2)
    nrows = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if len(names) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, name in enumerate(names):
        ax   = axes[idx // ncols][idx % ncols]
        data = taxonomy[name]
        labels = list(data.keys())
        sizes  = list(data.values())
        if not sizes:
            ax.set_visible(False)
            continue
        colors = plt.cm.Set3.colors[:len(labels)]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               colors=colors, pctdistance=0.82, startangle=120)
        ax.set_title(f'{name}', fontsize=12, pad=10)

    fig.suptitle('Error Taxonomy by Tagger', fontsize=15, y=1.01)
    plt.tight_layout()

    return _save(fig, os.path.join(output_dir, 'error_taxonomy.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 9. Cross-Tagger Error Category Breakdown
# ──────────────────────────────────────────────────────────────────────────────

def plot_error_category_breakdown(
    breakdown: Dict[str, int],
    output_dir: str
) -> str:
    categories = ['all_correct', 'only_one_wrong', 'majority_wrong', 'all_wrong']
    labels     = ['All Correct', 'Only 1 Wrong', 'Majority Wrong', 'All Wrong']
    values     = [breakdown.get(c, 0) for c in categories]
    colors     = ['#388E3C', '#FBC02D', '#F57C00', '#D32F2F']

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.88,
                  edgecolor='white', linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Number of Tokens')
    ax.set_title('Cross-Tagger Error Category Breakdown\n(how many taggers got each token wrong)')
    ax.grid(axis='y', alpha=0.3)

    return _save(fig, os.path.join(output_dir, 'error_category_breakdown.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 10. Ensemble vs Individual Comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_ensemble_comparison(
    individual_acc: Dict[str, float],
    ensemble_acc: float,
    output_dir: str
) -> str:
    names  = list(individual_acc.keys()) + ['Ensemble\n(Majority Vote)']
    values = [v * 100 for v in individual_acc.values()] + [ensemble_acc * 100]
    colors = PALETTE[:len(individual_acc)] + ['#FFB300']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, alpha=0.88,
                  edgecolor='white', linewidth=1.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f'{val:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylim(max(0, min(values) - 4), 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Individual Taggers vs Majority-Vote Ensemble')
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Highlight improvement
    best_ind = max(individual_acc.values()) * 100
    if ensemble_acc * 100 > best_ind:
        ax.annotate(
            f'+{ensemble_acc*100 - best_ind:.2f}% over best individual',
            xy=(len(names) - 1, ensemble_acc * 100),
            xytext=(len(names) - 2, ensemble_acc * 100 + 1),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='#555'
        )

    return _save(fig, os.path.join(output_dir, 'ensemble_comparison.png'))


# ──────────────────────────────────────────────────────────────────────────────
# 11. Sentence-Position Error Rates
# ──────────────────────────────────────────────────────────────────────────────

def plot_position_analysis(
    position_data: Dict[str, Dict[str, float]],
    output_dir: str
) -> str:
    """position_data: {tagger_name: {bucket: error_rate}}"""
    tagger_names = list(position_data.keys())
    if not tagger_names:
        return ''

    buckets = ['start (0–25%)', 'early-mid (25–50%)',
               'late-mid (50–75%)', 'end (75–100%)']
    x     = np.arange(len(buckets))
    width = 0.8 / len(tagger_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(tagger_names):
        rates  = [position_data[name].get(b, 0) * 100 for b in buckets]
        offset = (i - len(tagger_names) / 2 + 0.5) * width
        ax.bar(x + offset, rates, width * 0.9,
               label=name, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xlabel('Position in Sentence')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate by Token Position in Sentence')
    ax.set_xticks(x)
    ax.set_xticklabels(['Start\n(0–25%)', 'Early-Mid\n(25–50%)',
                         'Late-Mid\n(50–75%)', 'End\n(75–100%)'])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    return _save(fig, os.path.join(output_dir, 'position_error_rates.png'))
