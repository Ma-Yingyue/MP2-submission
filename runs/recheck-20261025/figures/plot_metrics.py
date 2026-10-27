import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


OUTPUT = Path(__file__).resolve().parent
INPUT = OUTPUT.parent
SPLITS = ('validation', 'test')
COLORS = ('#B4C0E4', '#484878')
TEXT = '#272727'
SECONDARY = '#606060'
FEATURE_COLORS = ('#7884B4', '#484878', '#B77794')
METRIC_LABELS = ('All validation groups', 'Answer-disagreement groups')


def load_metrics():
    records = []
    fingerprints = set()
    for split in SPLITS:
        score = json.loads((INPUT / f'{split}-score.json').read_text(encoding='utf-8'))
        resource = json.loads((INPUT / f'{split}-resources.json').read_text(encoding='utf-8'))
        if score['split'] != split or resource['split'] != split or not resource['completed']:
            raise ValueError(f'Incomplete or mismatched evaluation: {split}')
        if abs(score['accuracy'] - 100.0 * score['correct'] / score['n']) > 1e-9:
            raise ValueError(f'Accuracy and question counts disagree: {split}')
        processes = resource['processes']
        if len(processes) != 2 or any(process['returncode'] != 0 for process in processes):
            raise ValueError(f'Expected successful prediction and scoring processes: {split}')
        seconds = resource['complete_cpu_evaluation_seconds']
        process_seconds = [process['wall_seconds'] for process in processes]
        overhead = seconds - sum(process_seconds)
        if seconds <= 0 or overhead < -1e-6:
            raise ValueError(f'Inconsistent process timing: {split}')
        fingerprints.add(resource['checkpoint_fingerprint'])
        records.append({
            'split': split,
            'correct': score['correct'],
            'questions': score['n'],
            'accuracy_percent': score['accuracy'],
            'complete_cpu_evaluation_seconds': seconds,
            'prediction_process_seconds': process_seconds[0],
            'scoring_process_seconds': process_seconds[1],
            'orchestration_seconds': max(overhead, 0.0),
            'throughput_questions_per_second': score['n'] / seconds,
            'peak_observed_working_set_mib': resource['peak_observed_child_working_set_bytes'] / (1024 ** 2),
            'asset_bytes': resource['asset_bytes'],
            'asset_limit_bytes': resource['asset_limit_bytes'],
            'reference_cpu_verified': resource['measurement']['reference_cpu_verified'],
            'timing_measurements': 1,
        })
    if len(fingerprints) != 1:
        raise ValueError('The evaluations use different frozen checkpoints')
    if len({row['asset_bytes'] for row in records}) != 1:
        raise ValueError('Inference asset sizes differ between evaluations')
    return records


def load_experiments():
    path = INPUT.parent / 'final' / 'experiments.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    experiments = record['experiments']
    names = [experiment['name'] for experiment in experiments]
    if len(names) != len(set(names)):
        raise ValueError('Experiment names must be unique')
    for experiment in experiments:
        metric = experiment['validation']
        for numerator, denominator, percent in (
            ('correct', 'n', 'accuracy'),
            ('hard_correct', 'hard_n', 'hard_accuracy'),
        ):
            value = 100.0 * metric[numerator] / metric[denominator]
            if abs(value - metric[percent]) > 1e-9:
                raise ValueError(f'Experiment accuracy disagrees with counts: {experiment["name"]}')
    return experiments, record['selected']['name']


def experiment_rows(experiments):
    rows = []
    for experiment in experiments:
        metric = experiment['validation']
        parameters = experiment.get('parameters', {})
        rows.append({
            'experiment': experiment['name'],
            'validation_correct': metric['correct'],
            'validation_questions': metric['n'],
            'validation_accuracy_percent': metric['accuracy'],
            'disagreement_correct': metric['hard_correct'],
            'disagreement_questions': metric['hard_n'],
            'disagreement_accuracy_percent': metric['hard_accuracy'],
            'model_fitting_seconds': experiment['training_seconds'],
            'C': parameters.get('C'),
            'disagreement_only_training': parameters.get('hard_training_only'),
            'center_training_groups': parameters.get('center_training_groups'),
            'feature_modes': ','.join(component['mode'] for component in experiment.get('components', [])),
        })
    return rows


def style_axis(axis, ylabel=None, xlabel=None, grid='y'):
    axis.set_axisbelow(True)
    axis.grid(axis=grid, color='#E8E8EC', linewidth=0.5)
    if ylabel:
        axis.set_ylabel(ylabel, labelpad=6)
    if xlabel:
        axis.set_xlabel(xlabel, labelpad=6)
    axis.tick_params(axis='both', length=3, width=0.65, pad=4)
    axis.spines['left'].set_color('#707078')
    axis.spines['bottom'].set_color('#707078')


def new_figure(columns=1, height_mm=80, width_mm=183):
    figure, axes = plt.subplots(1, columns, figsize=(width_mm / 25.4, height_mm / 25.4),
                                squeeze=False, layout='constrained')
    figure.get_layout_engine().set(w_pad=0.07, h_pad=0.07, wspace=0.14)
    return figure, axes[0]


def save_figure(figure, name, rows):
    if figure.texts or figure._suptitle is not None:
        raise ValueError(f'Unexpected figure title or annotation: {name}')
    for axis in figure.axes:
        if axis.texts or any(axis.get_title(loc) for loc in ('left', 'center', 'right')):
            raise ValueError(f'Unexpected panel title or annotation: {name}')
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas_box = figure.bbox
    texts = []
    for axis in figure.axes:
        for positions, labels, limits in (
            (axis.get_xticks(), axis.get_xticklabels(), axis.get_xlim()),
            (axis.get_yticks(), axis.get_yticklabels(), axis.get_ylim()),
        ):
            lower, upper = sorted(limits)
            texts.extend(label for position, label in zip(positions, labels) if lower <= position <= upper)
        texts.extend((axis.xaxis.label, axis.yaxis.label))
        legend = axis.get_legend()
        if legend is not None:
            texts.extend(legend.get_texts())
    for item in texts:
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if box.x0 < canvas_box.x0 or box.y0 < canvas_box.y0 or box.x1 > canvas_box.x1 or box.y1 > canvas_box.y1:
                raise ValueError(f'Text extends beyond figure: {item.get_text()}')

    stem = OUTPUT / name
    figure.savefig(stem.with_suffix('.png'), dpi=600)
    figure.savefig(stem.with_suffix('.svg'))
    figure.savefig(stem.with_suffix('.pdf'))
    with (OUTPUT / f'{name}_source.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(figure)
    return name


def split_bars(axis, records, field, ylabel, upper=None):
    positions = np.arange(len(records))
    axis.bar(positions, [row[field] for row in records], width=0.55,
             color=COLORS, linewidth=0, zorder=3)
    axis.set_xticks(positions, [row['split'].title() for row in records])
    axis.set_xlim(-0.65, len(records) - 0.35)
    axis.set_ylim(bottom=0, top=upper)
    axis.tick_params(axis='x', length=0, pad=7)
    style_axis(axis, ylabel)


def plot_selection(records):
    figure, axes = new_figure(2)
    split_bars(axes[0], records, 'accuracy_percent', 'Selection accuracy (%)', 100)
    axes[0].yaxis.set_major_locator(MultipleLocator(20))
    split_bars(axes[1], records, 'complete_cpu_evaluation_seconds', 'Local complete CPU time (s)', 4)
    axes[1].yaxis.set_major_locator(MultipleLocator(1))
    return save_figure(figure, 'selection_metrics', records)


def plot_runtime(records):
    figure, axes = new_figure(width_mm=120)
    axis = axes[0]
    positions = np.arange(len(records))
    prediction = np.array([row['prediction_process_seconds'] for row in records])
    scoring = np.array([row['scoring_process_seconds'] for row in records])
    axis.bar(positions, prediction, width=0.55, color=COLORS[0], label='Prediction process', zorder=3)
    axis.bar(positions, scoring, bottom=prediction, width=0.55, color=COLORS[1],
             label='Scoring process', zorder=3)
    axis.set_xticks(positions, [row['split'].title() for row in records])
    axis.set_xlim(-0.65, len(records) - 0.35)
    axis.set_ylim(0, 4)
    axis.yaxis.set_major_locator(MultipleLocator(1))
    style_axis(axis, 'Local process wall time (s)')
    axis.legend(loc='upper right')
    return save_figure(figure, 'runtime_breakdown', records)


def plot_efficiency(records):
    figure, axes = new_figure(2)
    split_bars(axes[0], records, 'throughput_questions_per_second',
               'Local complete throughput (questions/s)', 300)
    split_bars(axes[1], records, 'peak_observed_working_set_mib',
               'Peak observed working set (MiB)', 110)
    return save_figure(figure, 'resource_efficiency', records)


def plot_assets(records):
    resource = json.loads((INPUT / 'test-resources.json').read_text(encoding='utf-8'))
    assets = resource['assets']
    rows = [{
        'asset': 'Configuration' if name == 'config.json' else 'Weights',
        'bytes': value['bytes'],
        'kibibytes': value['bytes'] / 1024,
    } for name, value in assets.items()]
    if sum(row['bytes'] for row in rows) != records[0]['asset_bytes']:
        raise ValueError('Asset components do not add up to the inference bundle size')
    figure, axes = new_figure(height_mm=68, width_mm=120)
    axis = axes[0]
    positions = np.arange(len(rows))
    axis.barh(positions, [row['kibibytes'] for row in rows], height=0.5, color=COLORS, zorder=3)
    axis.set_yticks(positions, [row['asset'] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 12)
    style_axis(axis, xlabel='Inference asset size (KiB)', grid='x')
    return save_figure(figure, 'asset_footprint', rows)


def experiment_label(experiment):
    name = experiment['name']
    labels = {
        'majority_valid_answer': 'Majority vote',
        'raw_embedding_all': 'Raw embeddings, all train',
        'centered_embedding_all': 'Centered embeddings, all train',
        'engineered_only_hard': 'Engineered features, linear',
        'engineered_boost_l7_m80': 'Boosting, L = 7, M = 80',
        'engineered_boost_l15_m80': 'Boosting, L = 15, M = 80',
        'engineered_boost_l15_m160': 'Boosting, L = 15, M = 160',
        'engineered_ranker': 'Ranking trees',
    }
    if name in labels:
        return labels[name]
    for prefix, label in (
        ('centered_embedding_hard_c', 'Centered embeddings'),
        ('local_hard_c', 'Embeddings + local'),
        ('full_hard_c', 'Embeddings + all'),
    ):
        if name.startswith(prefix):
            return f'{label}, C = {experiment["parameters"]["C"]:g}'
    raise ValueError(f'Unexpected experiment label: {name}')


def plot_method_comparison(experiments):
    methods = [experiment for experiment in experiments if '_fusion_' not in experiment['name']]
    rows = experiment_rows(methods)
    figure, axes = new_figure(height_mm=135)
    axis = axes[0]
    positions = np.arange(len(methods))
    full = [experiment['validation']['accuracy'] for experiment in methods]
    hard = [experiment['validation']['hard_accuracy'] for experiment in methods]
    axis.hlines(positions, hard, full, color='#D8D8E4', linewidth=1, zorder=2)
    axis.scatter(full, positions, color=COLORS[0], marker='o', edgecolors='#7884B4',
                 linewidths=0.5, s=22, label=METRIC_LABELS[0], zorder=3)
    axis.scatter(hard, positions, color=COLORS[1], marker='s', edgecolors=COLORS[1],
                 linewidths=0.5, s=22, label=METRIC_LABELS[1], zorder=3)
    axis.set_yticks(positions, [experiment_label(experiment) for experiment in methods])
    axis.tick_params(axis='y', labelsize=6.6, length=0)
    axis.set_ylim(len(methods) - 0.4, -0.9)
    axis.set_xlim(60, 100)
    axis.xaxis.set_major_locator(MultipleLocator(10))
    style_axis(axis, xlabel='Validation selection accuracy (%)', grid='x')
    axis.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.4)
    return save_figure(figure, 'validation_method_comparison', rows)


def grouped_accuracy(axis, methods, labels, xlabel):
    positions = np.arange(len(methods))
    for offset, field, color, legend in zip(
        (-0.18, 0.18), ('accuracy', 'hard_accuracy'), COLORS, METRIC_LABELS
    ):
        axis.bar(positions + offset, [method['validation'][field] for method in methods],
                 width=0.34, color=color, label=legend, linewidth=0, zorder=3)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 100)
    axis.yaxis.set_major_locator(MultipleLocator(20))
    style_axis(axis, 'Validation selection accuracy (%)', xlabel)
    axis.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)


def plot_feature_ablation(by_name):
    names = ('centered_embedding_hard_c0.1', 'local_hard_c0.1',
             'full_hard_c0.1', 'engineered_only_hard')
    methods = [by_name[name] for name in names]
    for method in methods:
        parameters = method['parameters']
        if parameters['C'] != 0.1 or not parameters['hard_training_only'] or not parameters['center_training_groups']:
            raise ValueError('Feature comparison requires matching training settings')
    figure, axes = new_figure(height_mm=87)
    grouped_accuracy(axes[0], methods,
                     ('Embeddings', 'Embeddings\n+ local features', 'Embeddings\n+ all features', 'Engineered\nfeatures'),
                     'Feature set (C = 0.1; answer-disagreement training groups)')
    return save_figure(figure, 'feature_ablation', experiment_rows(methods))


def plot_training_ablation(by_name):
    methods = [by_name['centered_embedding_all'], by_name['centered_embedding_hard_c0.1']]
    settings = [(method['parameters']['C'], method['parameters']['center_training_groups']) for method in methods]
    if settings[0] != settings[1]:
        raise ValueError('Training-group comparison requires identical C and centering')
    figure, axes = new_figure(height_mm=82, width_mm=140)
    grouped_accuracy(axes[0], methods, ('All training groups', 'Answer-disagreement\ntraining groups'),
                     'Training groups (centered embeddings, C = 0.1)')
    return save_figure(figure, 'training_group_ablation', experiment_rows(methods))


def plot_regularization(by_name):
    figure, axes = new_figure(2, height_mm=88)
    rows = []
    for prefix, label, color, marker in zip(
        ('centered_embedding_hard_c', 'local_hard_c', 'full_hard_c'),
        ('Embeddings', '+ local features', '+ all features'),
        FEATURE_COLORS, ('o', 's', '^'),
    ):
        methods = [by_name[f'{prefix}{value:g}'] for value in (0.01, 0.03, 0.1)]
        rows.extend(experiment_rows(methods))
        values = [method['parameters']['C'] for method in methods]
        for axis, field in zip(axes, ('accuracy', 'hard_accuracy')):
            axis.plot(values, [method['validation'][field] for method in methods],
                      color=color, marker=marker, markersize=4, linewidth=1, label=label)
    for axis, label, limits in zip(axes,
        ('Validation accuracy (%)', 'Validation disagreement accuracy (%)'), ((88, 96), (74, 88))
    ):
        axis.set_xscale('log')
        axis.set_xticks((0.01, 0.03, 0.1), ('0.01', '0.03', '0.1'))
        axis.set_xticks([], minor=True)
        axis.set_ylim(*limits)
        style_axis(axis, label, 'C')
        axis.legend(loc='lower left', fontsize=6.5)
    return save_figure(figure, 'regularization_sensitivity', rows)


def plot_fusion(by_name):
    figure, axes = new_figure(2, height_mm=88)
    base = by_name['local_hard_c0.01']
    rows = []
    families = ('engineered_boost_l7_m80', 'engineered_boost_l15_m80',
                'engineered_boost_l15_m160', 'engineered_ranker')
    labels = ('Boosting, L7 / M80', 'Boosting, L15 / M80', 'Boosting, L15 / M160', 'Ranking trees')
    for family, label, color, marker in zip(families, labels,
        ('#7884B4', '#484878', '#B77794', '#42949E'), ('o', 's', '^', 'D')
    ):
        weights = (0.0, 0.1, 0.2, 0.35, 0.5)
        methods = [base] + [by_name[f'local_hard_c0.01_{family}_fusion_{weight:g}'] for weight in weights[1:]]
        for weight, row in zip(weights, experiment_rows(methods)):
            rows.append({'fusion_component': family, 'fusion_weight': weight, **row})
        for axis, field in zip(axes, ('accuracy', 'hard_accuracy')):
            axis.plot(weights, [method['validation'][field] for method in methods],
                      color=color, marker=marker, markersize=3.6, linewidth=1, label=label)
    for axis, label, limits in zip(axes,
        ('Validation accuracy (%)', 'Validation disagreement accuracy (%)'), ((90, 95), (79, 88))
    ):
        axis.set_xticks((0, 0.1, 0.2, 0.35, 0.5), ('0', '0.1', '0.2', '0.35', '0.5'))
        axis.set_ylim(*limits)
        style_axis(axis, label, 'Tree component weight')
        axis.legend(loc='lower left', fontsize=6.2)
    return save_figure(figure, 'fusion_sensitivity', rows)


def model_family(experiment):
    if experiment['name'] == 'engineered_ranker':
        return 'Ranking trees'
    if experiment['components'][0]['kind'] == 'lightgbm':
        return 'Gradient boosting'
    return {
        'embedding': 'Linear: embeddings',
        'local': 'Linear: embeddings + local',
        'full': 'Linear: embeddings + all',
        'text': 'Linear: engineered features',
    }[experiment['components'][0]['mode']]


def plot_training_cost(experiments, selected):
    methods = [experiment for experiment in experiments
               if '_fusion_' not in experiment['name'] and experiment['training_seconds'] > 0]
    figure, axes = new_figure(height_mm=90)
    axis = axes[0]
    families = ('Linear: embeddings', 'Linear: embeddings + local', 'Linear: embeddings + all',
                'Linear: engineered features', 'Gradient boosting', 'Ranking trees')
    for family, color, marker in zip(families,
        ('#7884B4', '#484878', '#B77794', '#A8A8A8', '#42949E', '#9A4D8E'),
        ('o', 's', '^', 'D', 'v', 'P')
    ):
        members = [method for method in methods if model_family(method) == family]
        axis.scatter([method['training_seconds'] for method in members],
                      [method['validation']['hard_accuracy'] for method in members],
                      s=30, color=color, marker=marker, label=family, zorder=3)
    frozen = next(method for method in methods if method['name'] == selected)
    axis.scatter([frozen['training_seconds']], [frozen['validation']['hard_accuracy']],
                  s=85, marker='*', facecolors='none', edgecolors=TEXT, linewidths=0.8,
                  label='Frozen model', zorder=4)
    axis.set_xscale('log')
    axis.set_xlim(0.04, 10)
    axis.set_ylim(75, 88)
    style_axis(axis, 'Validation disagreement accuracy (%)', 'Model fitting time (s)')
    axis.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6.6)
    return save_figure(figure, 'training_cost', experiment_rows(methods))


def main():
    records = load_metrics()
    experiments, selected = load_experiments()
    by_name = {experiment['name']: experiment for experiment in experiments}
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 7,
        'text.color': TEXT,
        'axes.labelcolor': TEXT,
        'axes.labelsize': 7.5,
        'axes.linewidth': 0.65,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': TEXT,
        'ytick.color': TEXT,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    outputs = [
        plot_selection(records),
        plot_runtime(records),
        plot_efficiency(records),
        plot_assets(records),
        plot_method_comparison(experiments),
        plot_feature_ablation(by_name),
        plot_training_ablation(by_name),
        plot_regularization(by_name),
        plot_fusion(by_name),
        plot_training_cost(experiments, selected),
    ]
    print(json.dumps({
        'figures': outputs,
        'formats': ['png', 'svg', 'pdf'],
        'png_dpi': 600,
        'figure_count': len(outputs),
        'titles': False,
        'annotations': False,
        'text_within_canvas': True,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
