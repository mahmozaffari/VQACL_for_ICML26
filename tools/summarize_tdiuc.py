import json
import os
import re
import numpy as np
from argparse import ArgumentParser

task_size = {
    'absurd': 120411,
    'activity_recognition': 2682,
    'attribute': 9200,
    'color': 62490,
    'counting': 52905,
    'object_presence': 215324,
    'object_recognition': 215324,
    'positional_reasoning': 12284,
    'scene_recognition': 22032,
    'sentiment_understanding': 634,
    'sport_recognition': 10042,
    'utility_affordance': 171,
    'animal': 71698,
    'food': 81302,
    'indoor_activity': 61476,
    'outdoor_activity': 47704,
    'traffic': 51617,
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', type=str, required=True, help='Path to results JSON file or directory')
    parser.add_argument('-o', type=str, default=None, help='Output file for metrics (optional)')
    parser.add_argument('--task-order', type=str, default=None,
                        help='Comma-separated task order for BWT/forget (optional)')
    parser.add_argument('--last-task', type=str, default='q_causal',
                        help='Last task name for task-id accuracy aggregation (optional)')
    return parser.parse_args()

def backward_transfer(results, tasks):
    """
    Calculate backward transfer (BWT): how much previous tasks improve/degrade
    after training on all tasks compared to when they were just learned.
    """
    try:
        print('Calculating backward transfer...')
        # Get diagonal values (accuracy when task was just trained)
        Sii = []
        for task in tasks[:-1]:
            Sii.append(results[task][task])

        # Get final accuracies for all tasks except the last
        SiT = [results[tasks[-1]][task] for task in tasks[:-1]]

        assert len(Sii) == len(SiT) == len(tasks) - 1
        bwt = np.mean([SiT[i] - Sii[i] for i in range(len(SiT))])
    except Exception as e:
        print(f'Error calculating BWT: {e}')
        bwt = None
    return bwt

def average_forget(results, tasks):
    """
    Calculate average forgetting: difference between best performance on each task
    and final performance.
    """
    try:
        print('Calculating average forget...')
        # Find maximum accuracy achieved for each task
        Si_max = []
        for i in range(len(tasks) - 1):
            Si_max.append(max([results[task][tasks[i]] for task in tasks
                              if tasks[i] in results[task]]))

        # Get final accuracies for all tasks except the last
        SiT = [results[tasks[-1]][task] for task in tasks[:-1]]

        assert len(Si_max) == len(SiT) == len(tasks) - 1

        forget = np.mean([Si_max[i] - SiT[i] for i in range(len(SiT))])
    except Exception as e:
        print(f'Error calculating forget: {e}')
        forget = None

    return forget

def average_acc(results, tasks):
    """
    Calculate simple average accuracy across all tasks after full training.
    """
    acc = [results[tasks[-1]][task] for task in tasks]
    return np.mean(acc)

def total_acc(results, tasks):
    """
    Calculate weighted average accuracy based on task sizes.
    """
    if not all([task in task_size for task in tasks]):
        print('Warning: Task size not found for some tasks')
        return None

    total = 0
    num = 0
    for task in tasks:
        total += results[tasks[-1]][task] * task_size[task]
        num += task_size[task]
    return total / num

def _expected_result_files():
    return [
        'standard_accuracy.json',
        'standard_ece.json',
        'bayesian_accuracy.json',
        'bayesian_ece.json',
        'oracle_accuracy.json',
        'oracle_ece.json',
    ]

def _collect_result_files(results_path):
    if os.path.isdir(results_path):
        files = []
        for name in _expected_result_files():
            path = os.path.join(results_path, name)
            if os.path.isfile(path):
                files.append(path)
        return files
    return [results_path]

def _compute_performance(results, include_transfer=True, tasks_override=None):
    if tasks_override:
        tasks = [task for task in tasks_override if task in results]
    else:
        tasks = list(results.keys())
    bwt = backward_transfer(results, tasks) if include_transfer else None
    forget = average_forget(results, tasks) if include_transfer else None
    acc = average_acc(results, tasks)
    total = total_acc(results, tasks)
    performance = {
        'bwt': bwt,
        'forget': forget,
        'average_acc': acc,
        'weighted_acc': total
    }
    return performance

def get_metrics(results_path):
    """
    Load results and compute all metrics.
    """
    files = _collect_result_files(results_path)
    if len(files) == 1 and not os.path.isdir(results_path):
        with open(files[0], 'r') as f:
            results = json.load(f)
        return _compute_performance(results)

    performance_by_file = {}
    for path in files:
        with open(path, 'r') as f:
            results = json.load(f)
        performance_by_file[os.path.basename(path)] = _compute_performance(results)
    return performance_by_file

def _split_metric_name(filename):
    stem = filename.replace('.json', '')
    if stem.endswith('_accuracy'):
        return stem[:-9], 'accuracy'
    if stem.endswith('_ece'):
        return stem[:-4], 'ece'
    return stem, 'accuracy'

def _find_predictions_dir(results_dir, group, last_task=None):
    candidate = os.path.join(results_dir, 'predictions', group)
    if os.path.isdir(candidate):
        return candidate
    if os.path.basename(results_dir) == group and os.path.isdir(results_dir):
        return results_dir
    found = []
    for root, dirs, _ in os.walk(results_dir):
        if os.path.basename(root) == 'predictions' and group in dirs:
            found.append(os.path.join(root, group))
    if not found:
        return None
    if last_task:
        pattern = re.compile(rf'^after_{re.escape(last_task)}_on_.+\\.json$')
        for path in sorted(found, key=lambda p: os.path.getmtime(p), reverse=True):
            try:
                if any(pattern.match(name) for name in os.listdir(path)):
                    return path
            except Exception:
                continue
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found[0]

def _compute_task_id_metrics(predictions_dir, last_task):
    if not predictions_dir or not last_task:
        return None
    pattern = re.compile(r'^after_(.+)_on_(.+)\.json$')
    values = []
    total = 0.0
    weight = 0
    for name in os.listdir(predictions_dir):
        match = pattern.match(name)
        if not match:
            continue
        train_task, test_task = match.group(1), match.group(2)
        if train_task != last_task:
            continue
        path = os.path.join(predictions_dir, name)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        value = data.get('task_id_accuracy')
        if value is None:
            value = (data.get('additional_metrics') or {}).get('task_id_accuracy') * 100.0
        if value is None:
            continue
        values.append(value)
        if test_task in task_size:
            total += value * task_size[test_task]
            weight += task_size[test_task]
    if not values:
        return None
    avg = float(np.mean(values))
    weighted = (total / weight) if weight > 0 else None
    return {'avg': avg, 'weighted': weighted, 'count': len(values)}

def _format_metrics_table(performance_by_file, task_id_metrics_by_group=None):
    headers = [
        'group',
        'acc_bwt',
        'acc_forget',
        'acc_avg',
        'acc_weighted',
        'taskid_avg',
        'taskid_weighted',
        'ece_avg',
        'ece_weighted',
    ]
    grouped = {}
    for name, metrics in performance_by_file.items():
        group, kind = _split_metric_name(name)
        grouped.setdefault(group, {})[kind] = metrics

    rows = []
    for group in sorted(grouped.keys()):
        acc = grouped[group].get('accuracy')
        ece = grouped[group].get('ece')
        taskid_avg = 'N/A'
        taskid_weighted = 'N/A'
        task_id_metrics = (task_id_metrics_by_group or {}).get(group)
        if task_id_metrics:
            taskid_avg = f"{task_id_metrics['avg']:.4f}"
            taskid_weighted = 'N/A' if task_id_metrics['weighted'] is None else f"{task_id_metrics['weighted']:.4f}"
        row = [
            group,
            'N/A' if not acc or acc['bwt'] is None else f"{acc['bwt']:.4f}",
            'N/A' if not acc or acc['forget'] is None else f"{acc['forget']:.4f}",
            'N/A' if not acc else f"{acc['average_acc']:.4f}",
            'N/A' if not acc or acc['weighted_acc'] is None else f"{acc['weighted_acc']:.4f}",
            taskid_avg,
            taskid_weighted,
            'N/A' if not ece else f"{ece['average_acc']:.4f}",
            'N/A' if not ece or ece['weighted_acc'] is None else f"{ece['weighted_acc']:.4f}",
        ]
        rows.append(row)

    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]
    lines = []
    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "  ".join('-' * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)

if __name__ == '__main__':
    args = parse_args()
    tasks_override = None
    if args.task_order:
        tasks_override = [t.strip() for t in args.task_order.split(',') if t.strip()]

    if os.path.isdir(args.f):
        files = _collect_result_files(args.f)
        if not files:
            print(f'No matching result files found in directory: {args.f}')
            raise SystemExit(1)
        performance_by_file = {}
        for path in files:
            include_transfer = not os.path.basename(path).endswith('_ece.json')
            with open(path, 'r') as f:
                results = json.load(f)
            performance_by_file[os.path.basename(path)] = _compute_performance(
                results,
                include_transfer,
                tasks_override,
            )
        task_id_metrics_by_group = {}
        for group in ('standard', 'bayesian', 'oracle'):
            predictions_dir = _find_predictions_dir(args.f, group, args.last_task)
            task_id_metrics = _compute_task_id_metrics(predictions_dir, args.last_task)
            if task_id_metrics:
                task_id_metrics_by_group[group] = task_id_metrics

        print('\n' + '='*70)
        print('CONTINUAL LEARNING METRICS (DIRECTORY)')
        print('='*70)
        print(_format_metrics_table(performance_by_file, task_id_metrics_by_group))
        missing = [name for name in _expected_result_files()
                   if not os.path.isfile(os.path.join(args.f, name))]
        if missing:
            print('\nMissing expected files:')
            for name in missing:
                print(f'  - {name}')
        if args.last_task and not task_id_metrics_by_group:
            print('\nTask-id accuracy: no matching prediction files found.')
        print('='*70)

        if args.o:
            with open(args.o, 'w') as f:
                json.dump(performance_by_file, f, indent=4)
            print(f'\nMetrics saved to {args.o}')
    else:
        with open(args.f, 'r') as f:
            results = json.load(f)

        tasks = tasks_override or list(results.keys())
        print(f'Tasks: {tasks}\n')

        include_transfer = not os.path.basename(args.f).endswith('_ece.json')
        performance = _compute_performance(results, include_transfer, tasks_override)
        bwt = performance['bwt']
        forget = performance['forget']
        acc = performance['average_acc']
        total = performance['weighted_acc']

        print('\n' + '='*50)
        print('CONTINUAL LEARNING METRICS')
        print('='*50)
        print(f'Backward Transfer:    {bwt:.4f}' if bwt is not None else 'Backward Transfer:    N/A')
        print(f'Average Forget:       {forget:.4f}' if forget is not None else 'Average Forget:       N/A')
        print(f'Average Accuracy:     {acc:.4f}')
        print(f'Weighted Accuracy:    {total:.4f}' if total is not None else 'Weighted Accuracy:    N/A')
        print('='*50)

        if args.o:
            metrics = {
                'backward_transfer': bwt,
                'average_forget': forget,
                'average_accuracy': acc,
                'weighted_accuracy': total
            }
            with open(args.o, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f'\nMetrics saved to {args.o}')
