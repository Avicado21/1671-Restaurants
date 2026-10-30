"""Reproduce Part 1 dataset figures from the small pipeline reports.

No model is trained, and no review CSV or feature matrix is read.
Use --project-dir for the Pitt files, or --source-dir for bundled reports.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

FEATURE_NAMES = {
    (1, 1): 'Unigrams', (1, 2): 'Unigrams + bigrams',
    (1, 3): 'Unigrams + bigrams + trigrams',
    (2, 2): 'Bigrams only', (3, 3): 'Trigrams only',
}


def parse_distribution(path):
    text = Path(path).read_text(encoding='utf-8')
    blocks = text.split('CAPPED VARIANT', maxsplit=1)
    if len(blocks) != 2:
        raise ValueError('Expected both NATURAL and CAPPED sections in the distribution report.')
    variants = {}
    for variant, block in zip(['natural', 'capped'], blocks):
        rows = {}
        for line in block.splitlines():
            match = re.match(r'^(.+?)\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d[\d,]*)\s+[\d.]+%\s*$', line)
            if not match:
                continue
            label = match.group(1).strip()
            values = [int(value.replace(',', '')) for value in match.groups()[1:]]
            if label in rows or sum(values[:3]) != values[3]:
                raise ValueError(f'Duplicate label or inconsistent row totals: {label}')
            rows[label] = dict(zip(['train', 'dev', 'test', 'total'], values))
        reported = re.search(r'^TOTAL\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s*$', block, re.M)
        if not rows or not reported:
            raise ValueError(f'Could not read {variant} counts.')
        for split, value in zip(['train', 'dev', 'test', 'total'], reported.groups()):
            if sum(row[split] for row in rows.values()) != int(value.replace(',', '')):
                raise ValueError(f'{variant} {split} does not match the printed total.')
        variants[variant] = rows
    if set(variants['natural']) != set(variants['capped']):
        raise ValueError('Natural and capped labels differ.')
    return variants


def parse_features(path, capped):
    text = Path(path).read_text(encoding='utf-8')
    if not re.search(r'^Variant:\s*capped\s*$', text, re.M):
        raise ValueError('This analysis expects the capped feature report.')
    matches = list(re.finditer(r'^\((\d+),(\d+)\)\s+[^\n]+$', text, re.M))
    records = []
    expected = {split: sum(row[split] for row in capped.values()) for split in ['train', 'dev', 'test']}
    for index, match in enumerate(matches):
        key = (int(match.group(1)), int(match.group(2)))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        vocab = re.search(r'vocabulary:\s*([\d,]+)\s+features', block)
        if not vocab or key not in FEATURE_NAMES:
            raise ValueError(f'Unrecognized feature section: {key}')
        n_features = int(vocab.group(1).replace(',', ''))
        for split, total in expected.items():
            shape = re.search(r'^\s*' + split + r'\s+([\d,]+)\s+x\s+([\d,]+)', block, re.M)
            if not shape or int(shape.group(1).replace(',', '')) != total or int(shape.group(2).replace(',', '')) != n_features:
                raise ValueError(f'{key} {split} shape disagrees with the dataset report.')
        records.append({'ngram_min': key[0], 'ngram_max': key[1],
                        'feature_set': FEATURE_NAMES[key], 'n_features': n_features})
    if len(records) != len(FEATURE_NAMES) or len({(r['ngram_min'], r['ngram_max']) for r in records}) != len(FEATURE_NAMES):
        raise ValueError('Expected all five unique n-gram feature sets.')
    return records


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_outputs(distribution_path, feature_stats_path, output_dir):
    distribution_path, feature_stats_path = Path(distribution_path), Path(feature_stats_path)
    variants = parse_distribution(distribution_path)
    natural, capped = variants['natural'], variants['capped']
    features = parse_features(feature_stats_path, capped)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {variant: {split: sum(row[split] for row in rows.values())
                       for split in ['train', 'dev', 'test', 'total']}
              for variant, rows in variants.items()}
    labels = sorted(capped)
    total_natural, total_capped = totals['natural']['total'], totals['capped']['total']
    cuisine_rows = []
    for label in labels:
        cuisine_rows.append({'cuisine': label,
                             'natural_total': natural[label]['total'],
                             'natural_share_pct': 100 * natural[label]['total'] / total_natural,
                             'capped_train': capped[label]['train'],
                             'capped_dev': capped[label]['dev'],
                             'capped_test': capped[label]['test'],
                             'capped_total': capped[label]['total'],
                             'capped_share_pct': 100 * capped[label]['total'] / total_capped})
    write_csv(out / 'cuisine_counts.csv', list(cuisine_rows[0]), cuisine_rows)
    split_rows = [{'split': split, 'natural_rows': totals['natural'][split],
                   'capped_rows': totals['capped'][split],
                   'capped_share_pct': 100 * totals['capped'][split] / total_capped}
                  for split in ['train', 'dev', 'test']]
    write_csv(out / 'split_counts.csv', list(split_rows[0]), split_rows)
    write_csv(out / 'feature_counts.csv', list(features[0]), features)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 16, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#27343E', 'axes.labelcolor': '#27343E'})
    navy, teal = '#42688E', '#168578'
    def finish(fig, filename):
        fig.savefig(out / filename, dpi=250, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    ordered = sorted(labels, key=lambda label: capped[label]['total'], reverse=True)
    values = [capped[label]['total'] for label in ordered]
    fig, ax = plt.subplots(figsize=(9.5, 7.2), layout='constrained')
    bars = ax.barh(ordered, values, color=teal, height=.65)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{value:,}' for value in values], padding=4, fontsize=10)
    ax.set_xlim(0, max(values) * 1.2)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Reviews across training, development, and test')
    ax.set_title('Final dataset reviews by cuisine', loc='left', pad=30)
    ax.text(0, 1.015, f'Capped dataset | {total_capped:,} reviews | {len(labels)} cuisines',
            transform=ax.transAxes, fontsize=11)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    finish(fig, '01_final_cuisine_counts.png')

    ordered = sorted(labels, key=lambda label: natural[label]['total'], reverse=True)
    natural_pct = np.array([100 * natural[label]['total'] / total_natural for label in ordered])
    capped_pct = np.array([100 * capped[label]['total'] / total_capped for label in ordered])
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9.5, 7.2), layout='constrained')
    ax.barh(y - .18, natural_pct, height=.34, color=navy, label=f'Natural ({total_natural:,} reviews)')
    ax.barh(y + .18, capped_pct, height=.34, color=teal, label=f'Capped ({total_capped:,} reviews)')
    ax.set_yticks(y, ordered)
    ax.invert_yaxis()
    ax.set_xlim(0, max(10, np.ceil(max(natural_pct.max(), capped_pct.max()) / 5) * 5))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel('Share of reviews within each dataset')
    ax.set_title('Cuisine balance before and after sampling', loc='left', pad=15)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    finish(fig, '02_natural_vs_capped.png')

    split_names = ['Training', 'Development', 'Test']
    split_values = [totals['capped'][split] for split in ['train', 'dev', 'test']]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    bars = ax.bar(split_names, split_values, color=[navy, teal, '#7A8997'], width=.55)
    ax.bar_label(bars, labels=[f'{value:,} ({100 * value / total_capped:.1f}%)' for value in split_values], padding=6)
    ax.set_ylim(0, max(split_values) * 1.2)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('Reviews')
    ax.set_title('Final dataset split sizes', loc='left', pad=14)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    finish(fig, '03_final_split_sizes.png')

    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    feature_names = [row['feature_set'] for row in features]
    feature_values = [row['n_features'] for row in features]
    bars = ax.barh(feature_names, feature_values, color=navy, height=.6)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{value:,}' for value in feature_values], padding=4)
    ax.set_xlim(0, max(feature_values) * 1.24)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Vocabulary features')
    ax.set_title('Vocabulary size of the prepared feature sets', loc='left', pad=14)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    finish(fig, '04_feature_vocabulary_sizes.png')

    summary = {'dataset': 'capped', 'rows': total_capped, 'columns': 9,
               'labels': len(labels), 'totals': totals,
               'capped_fraction_of_natural': total_capped / total_natural,
               'capped_smallest_class': min(row['total'] for row in capped.values()),
               'capped_largest_class': max(row['total'] for row in capped.values()),
               'features': features,
               'sources': [{'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in [distribution_path, feature_stats_path]],
               'scope': 'Dataset descriptions and distributions only; no classifier was trained or evaluated.'}
    (out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'FIGURE_NOTES.txt').write_text(
        '01_final_cuisine_counts.png: counts in the final capped dataset across all three splits.\n'
        '02_natural_vs_capped.png: shares are calculated separately within each dataset; the denominators differ.\n'
        '03_final_split_sizes.png: actual capped split sizes, not exact nominal 80/10/10 counts.\n'
        '04_feature_vocabulary_sizes.png: number of feature columns, not accuracy or model quality.\n'
        'The raw CSV has nine columns. The vectorized feature matrices have the vocabulary dimensions shown.\n',
        encoding='utf-8')
    print(f'Validated {len(labels)} cuisines; final dataset has {total_capped:,} reviews and 9 CSV columns.')
    print(f'Created four figures and supporting tables in {out.resolve()}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path('/ix1/cs1671-2026f/yelp'))
    parser.add_argument('--source-dir', type=Path, help='Use bundled, flat report files instead of the Pitt project.')
    parser.add_argument('--output-dir', type=Path, default=Path('part1_outputs'))
    args = parser.parse_args()
    if args.source_dir:
        distribution = args.source_dir / 'dataset_distribution.txt'
        feature_stats = args.source_dir / 'feature_stats.txt'
    else:
        distribution = args.project_dir / 'datasets/dataset_distribution.txt'
        feature_stats = args.project_dir / 'features/capped/feature_stats.txt'
    generate_outputs(distribution, feature_stats, args.output_dir)


if __name__ == '__main__':
    main()
