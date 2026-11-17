"""Verify the original training logs and plot their development scores.

Use --check for a standard-library-only evidence check. Rendering needs
matplotlib==3.11.2; no training runtime or model execution is required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURVES = HERE / 'curves'
PRIMARY = ('large-943', 'ppo-large-943', 'local-ppo-large-943')
GROUPS = (
    ('Imitation', ('large-941', 'large-942', 'large-943', 'large-944', 'script-941', 'script-942')),
    ('PPO refinement', ('ppo-large-942', 'ppo-large-943', 'ppo-script-942')),
    ('CMA-ES refinement', ('search-ppo-large-943', 'search-ppo-script-942', 'local-ppo-large-943', 'local-ppo-script-942')),
)


def read_curves():
    runs = json.loads((HERE / 'results.json').read_text())['runs']
    manifest = json.loads((CURVES / 'manifest.json').read_text())
    files = {path.stem for path in (CURVES / 'metrics').glob('*.jsonl')}
    if manifest['schema'] != 1 or set(manifest['runs']) != set(runs) or files != set(runs):
        raise ValueError('Expected the original logs for all 13 frozen runs.')
    curves = {}
    for name, record in runs.items():
        raw = (CURVES / 'metrics' / (name + '.jsonl')).read_bytes()
        source = manifest['runs'][name]
        if hashlib.sha256(raw).hexdigest() != source['metrics_sha256']:
            raise ValueError(f'{name}: metrics hash mismatch')
        if source['selected_model_sha256'] != record['run']['sha256']:
            raise ValueError(f'{name}: selected model differs from frozen evidence')
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != source['rows']:
            raise ValueError(f'{name}: incomplete log')
        search = name.startswith(('search-', 'local-'))
        coordinate = 'generation' if search else 'round'
        expected = (range(1, record['config']['generations'] + 1) if search else
                    range(record['config']['ppo']['rounds'] + 1) if name.startswith('ppo-') else
                    range(1, record['config']['rounds'] + 1))
        if [row[coordinate] for row in rows] != list(expected):
            raise ValueError(f'{name}: missing or repeated training steps')
        evaluated = [row for row in rows if 'development_score' in row]
        for row in evaluated:
            score = row['development_score']
            if len(score) != 2 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in score):
                raise ValueError(f'{name}: invalid development score')
            if not search:
                if set(row['opponents']) != {'attack', 'defend', 'interceptor', 'large'}:
                    raise ValueError(f'{name}: development opponent field changed')
                means = [sum(opponent[key] for opponent in row['opponents'].values()) / 4
                         for key in ('league_score', 'rally_score')]
                if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(score, means)):
                    raise ValueError(f'{name}: aggregate differs from opponent scores')
        best = max(row['development_score'] for row in evaluated)
        if best != record['run']['best_development_score'] or best != record['selection']['development_score']:
            raise ValueError(f'{name}: best checkpoint differs from frozen selection')
        selected = record['selection']
        selected_step = selected[coordinate]
        selected_row = next((row for row in evaluated if row[coordinate] == selected_step), None)
        if (not search or selected_step) and (selected_row is None or selected_row['development_score'] != best):
            raise ValueError(f'{name}: selected step does not match the log')
        if search and any(a['development_score'] > b['development_score'] for a, b in zip(evaluated, evaluated[1:])):
            raise ValueError(f'{name}: search incumbent score decreased')
        curves[name] = (record, evaluated)
    print(f'Verified all {len(curves)} logs against their hashes and frozen checkpoint selections.')
    return curves


def plot(curves, all_runs=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#bcc8c2', 'text.color': '#183a2d',
                         'axes.labelcolor': '#365348', 'xtick.color': '#52655b',
                         'ytick.color': '#52655b', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.3), sharey=True)
    fig.subplots_adjust(left=.065, right=.98, bottom=.36 if all_runs else .25, top=.73, wspace=.22)
    title = 'Every training run' if all_runs else 'More training did not always improve play'
    subtitle = ('Six imitation starts, three PPO refinements and four CMA-ES searches.' if all_runs else
                'The selected path to Small: 163 parameters, with the best checkpoints retained.')
    fig.text(.065, .94, title, fontsize=21, weight='bold')
    fig.text(.065, .885, subtitle, fontsize=11, color='#52655b')
    colors = dict(zip(GROUPS[0][1], ('#8a6eae', '#287ca3', '#176b4b', '#b19627', '#c26342', '#788791')))
    for panel, (ax, (title, names)) in enumerate(zip(axes, GROUPS)):
        search = panel == 2
        coordinate = 'generation' if search else 'round'
        if not all_runs:
            names = (PRIMARY[panel],)
        for name in names:
            record, rows = curves[name]
            x = [row[coordinate] for row in rows]
            y = [100 * row['development_score'][0] for row in rows]
            lineage = name.removeprefix('search-').removeprefix('local-').removeprefix('ppo-')
            color = colors[lineage] if all_runs else '#176b4b'
            if search:
                label = name.replace('search-ppo-', 'broad / ').replace('local-ppo-', 'local / ')
                ax.step(x, y, where='post', color=color, linewidth=2, label=label,
                        linestyle='--' if name.startswith('search-') else '-')
            else:
                label = name.removeprefix('ppo-')
                ax.plot(x, y, '.-', color=color, linewidth=1.6, markersize=5, label=label)
            selected = record['selection']
            sx, sy = selected[coordinate], 100 * selected['development_score'][0]
            ax.scatter(sx, sy, marker='D', s=55 if all_runs else 70,
                       color=color if all_runs else '#c26342', edgecolor='white', zorder=5)
            if not all_runs:
                ax.annotate(f'Retained: {sy:.1f}%\n{coordinate.capitalize()} {sx}', (sx, sy),
                            xytext=(8, 16), textcoords='offset points', fontsize=10, color='#9d442b')
                if not search:
                    ax.annotate(f'Final: {y[-1]:.1f}%', (x[-1], y[-1]), xytext=(-7, -23),
                                textcoords='offset points', ha='right', color='#176b4b', fontsize=10)
        ax.set_title(title + (' · best so far' if search else ' · current checkpoint'),
                     loc='left', fontsize=12, weight='bold', pad=13)
        ax.set_ylim(0, 40)
        ax.set_yticks(range(0, 41, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlim(-3 if panel != 1 else -8, 101 if panel != 1 else 270)
        ax.set_xticks([0, 24, 48, 72, 96] if panel != 1 else [0, 64, 128, 192, 256])
        ax.set_xlabel('Search generation' if search else 'Training round', labelpad=9)
        ax.grid(axis='y', color='#e6ece8', linewidth=.8)
        ax.set_axisbelow(True)
        if all_runs:
            ax.legend(loc='upper left', bbox_to_anchor=(0, -.25), frameon=False,
                      ncol=2 if panel != 2 else 1, fontsize=9, handlelength=1.7, columnspacing=1)
    axes[0].set_ylabel('Mean development paired-contest score', labelpad=10)
    notes = ('Development opponents: attack, defend, interceptor and Large; 32 fixed paired serve sets each. Quill and Counter were held out.\n'
             'Imitation/PPO: current checkpoints, evaluated every 8/16 rounds. CMA-ES: retained best-so-far score; a flat line does not show candidate stability.\n'
             'Diamonds mark retained checkpoints (rally score breaks ties). Stages have separate training budgets; curves are unsmoothed.')
    fig.text(.065, .055, notes, fontsize=9, color='#52655b', linespacing=1.7)
    output = CURVES / ('all-runs.png' if all_runs else 'selected-path.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f'Wrote {output.relative_to(HERE)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify evidence without rendering figures.')
    args = parser.parse_args()
    curves = read_curves()
    if not args.check:
        plot(curves)
        plot(curves, all_runs=True)


if __name__ == '__main__':
    main()
