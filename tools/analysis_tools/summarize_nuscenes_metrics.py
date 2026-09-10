#!/usr/bin/env python3
"""Export nuScenes detection metrics as thesis-friendly tables.

The nuScenes devkit writes the authoritative values to
``metrics_summary.json``.  This utility preserves those values in a compact
JSON file and creates CSV and Markdown tables suitable for later analysis.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


ERROR_NAMES = {
    'trans_err': 'mATE',
    'scale_err': 'mASE',
    'orient_err': 'mAOE',
    'vel_err': 'mAVE',
    'attr_err': 'mAAE',
}
CLASS_ERROR_KEYS = tuple(ERROR_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create thesis-ready tables from nuScenes metrics.')
    parser.add_argument('metrics_summary', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--validation-samples', type=int, default=6019)
    return parser.parse_args()


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def find_runtime(work_dir: Optional[Path]) -> Dict[str, Any]:
    runtime = {
        'data_time_s_per_sample': None,
        'inference_time_s_per_sample': None,
        'throughput_samples_per_s': None,
    }
    if work_dir is None or not work_dir.is_dir():
        return runtime

    metric_files = list(work_dir.glob('*/vis_data/scalars.json'))
    # Depending on the MMEngine version, LocalVisBackend writes either
    # ``vis_data/scalars.json`` or ``<timestamp>/<timestamp>.json``.
    metric_files.extend(
        path for path in work_dir.glob('*/*.json')
        if path.parent.name == path.stem)
    metric_files = sorted(
        metric_files,
        key=lambda path: path.stat().st_mtime,
        reverse=True)
    for scalar_file in metric_files:
        metric_record = None
        with scalar_file.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if any(key.endswith('/NDS') for key in record):
                    metric_record = record
        if metric_record is None:
            continue

        data_time = finite_or_none(metric_record.get('data_time'))
        inference_time = finite_or_none(metric_record.get('time'))
        runtime['data_time_s_per_sample'] = data_time
        runtime['inference_time_s_per_sample'] = inference_time
        if inference_time is not None and inference_time > 0:
            runtime['throughput_samples_per_s'] = 1.0 / inference_time
        runtime['source'] = str(scalar_file)
        break
    return runtime


def markdown_value(value: Optional[float]) -> str:
    return 'N/A' if value is None else f'{value:.4f}'


def write_csv_files(output_dir: Path, report: Dict[str, Any]) -> None:
    global_path = output_dir / 'metrics_global.csv'
    with global_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE',
                         'mAAE'])
        metrics = report['global_metrics']
        writer.writerow([metrics[key] for key in writer_header_global()])

    class_path = output_dir / 'metrics_per_class.csv'
    class_header = [
        'class', 'mean_AP', 'AP_0.5m', 'AP_1.0m', 'AP_2.0m', 'AP_4.0m',
        *CLASS_ERROR_KEYS
    ]
    with class_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=class_header)
        writer.writeheader()
        writer.writerows(report['per_class_metrics'])


def writer_header_global():
    return ('NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE')


def write_markdown(output_dir: Path, report: Dict[str, Any]) -> None:
    metrics = report['global_metrics']
    runtime = report['runtime']
    global_row = ' | '.join(
        markdown_value(metrics[key]) for key in writer_header_global())
    lines = [
        '# BEVFusion nuScenes validation metrics',
        '',
        f"- Checkpoint: `{report['checkpoint'] or 'not recorded'}`",
        f"- Evaluation split: nuScenes validation "
        f"({report['validation_samples']} samples)",
        '- Protocol: `detection_cvpr_2019` (official nuScenes detection)',
        '- Modalities: LiDAR + six surround-view cameras',
        f"- Authoritative source: `{report['metrics_summary_source']}`",
        '',
        '## Global metrics',
        '',
        '| NDS ↑ | mAP ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | '
        'mAVE ↓ | mAAE ↓ |',
        '|---:|---:|---:|---:|---:|---:|---:|',
        f'| {global_row} |',
        '',
        '## Per-class metrics',
        '',
        '| Class | mean AP ↑ | AP@0.5 m ↑ | AP@1.0 m ↑ | '
        'AP@2.0 m ↑ | AP@4.0 m ↑ | trans. err ↓ | scale err ↓ | '
        'orient. err ↓ | vel. err ↓ | attr. err ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['per_class_metrics']:
        values = [row['class'], *(
            markdown_value(row[key]) for key in (
                'mean_AP', 'AP_0.5m', 'AP_1.0m', 'AP_2.0m', 'AP_4.0m',
                *CLASS_ERROR_KEYS))]
        lines.append('| ' + ' | '.join(values) + ' |')

    lines.extend([
        '',
        '## Runtime',
        '',
        f"- Final-window data time: "
        f"{markdown_value(runtime['data_time_s_per_sample'])} s/sample",
        f"- Final-window test-loop time: "
        f"{markdown_value(runtime['inference_time_s_per_sample'])} s/sample",
        f"- Approximate throughput: "
        f"{markdown_value(runtime['throughput_samples_per_s'])} samples/s",
        '',
        'Runtime is the final MMEngine logger-window measurement and includes '
        'model inference plus loop overhead; it is not a '
        'hardware-independent model latency benchmark.',
        '',
        '## Metric interpretation',
        '',
        '- NDS and mAP: higher is better.',
        '- mATE (m), mASE (1 - scale IoU), mAOE (rad), mAVE (m/s), and '
        'mAAE (1 - attribute accuracy): lower is better.',
        '- AP is averaged over the four center-distance thresholds for the '
        'reported mAP.',
        '- `N/A` denotes a nuScenes metric that is not defined for that '
        'class.',
        '',
    ])
    (output_dir / 'metrics_thesis.md').write_text(
        '\n'.join(lines), encoding='utf-8')


def main() -> None:
    args = parse_args()
    with args.metrics_summary.open(encoding='utf-8') as stream:
        source = json.load(stream)

    global_metrics = {
        'NDS': finite_or_none(source['nd_score']),
        'mAP': finite_or_none(source['mean_ap']),
    }
    for source_name, report_name in ERROR_NAMES.items():
        global_metrics[report_name] = finite_or_none(
            source['tp_errors'].get(source_name))

    per_class = []
    for class_name, aps in source['label_aps'].items():
        errors = source['label_tp_errors'][class_name]
        mean_dist_aps = source.get('mean_dist_aps', {})
        mean_ap = mean_dist_aps.get(class_name)
        if mean_ap is None:
            mean_ap = sum(float(value) for value in aps.values()) / len(aps)
        per_class.append({
            'class': class_name,
            'mean_AP': finite_or_none(mean_ap),
            'AP_0.5m': finite_or_none(aps.get('0.5')),
            'AP_1.0m': finite_or_none(aps.get('1.0')),
            'AP_2.0m': finite_or_none(aps.get('2.0')),
            'AP_4.0m': finite_or_none(aps.get('4.0')),
            **{
                key: finite_or_none(errors.get(key))
                for key in CLASS_ERROR_KEYS
            },
        })

    report = {
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'evaluation_split': 'nuScenes validation',
        'validation_samples': args.validation_samples,
        'evaluation_protocol': 'detection_cvpr_2019',
        'modalities': {'use_lidar': True, 'use_camera': True},
        'metrics_summary_source': str(args.metrics_summary),
        'global_metrics': global_metrics,
        'per_class_metrics': per_class,
        'runtime': find_runtime(args.work_dir),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'metrics_thesis.json').open(
            'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    write_csv_files(args.output_dir, report)
    write_markdown(args.output_dir, report)
    print(f'Thesis metrics written to {args.output_dir}')


if __name__ == '__main__':
    main()
