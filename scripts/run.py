#!/usr/bin/env python3
"""Portable entrypoints for the local BEVFusion experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FUSION = 'bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13'
LIDAR = 'bevfusion_lidar_voxel0075_second_secfpn_1xb5-16e_nus-3d-quality'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train-lidar', 'train', 'validate', 'test', 'predict'])
    parser.add_argument('--checkpoint', default=f'work_dirs/{FUSION}/iter_741480.pth')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    os.chdir(ROOT)
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    config = f'projects/BEVFusion/configs/{LIDAR if args.action == "train-lidar" else FUSION}.py'
    if args.action.startswith('train'):
        command = [sys.executable, 'tools/train.py', config] + extra
    else:
        output = f'work_dirs/{FUSION}/{args.action}'
        command = [sys.executable, 'tools/test.py', config, args.checkpoint, '--work-dir', output]
        overrides = [f'test_evaluator.jsonfile_prefix={output}/nuscenes_results']
        if args.action == 'test':
            overrides += ['test_dataloader.dataset.ann_file=nuscenes_infos_test.pkl',
                          'test_dataloader.dataset.metainfo.version=v1.0-test',
                          'test_evaluator.ann_file=data/nuscenes/nuscenes_infos_test.pkl',
                          'test_evaluator.format_only=True']
        if args.action == 'predict':
            overrides += ['test_evaluator.format_only=True']
        # Merge caller overrides into a single DictAction occurrence.
        if '--cfg-options' in extra:
            idx = extra.index('--cfg-options')
            command += extra[:idx] + ['--cfg-options'] + overrides + extra[idx+1:]
        else:
            command += extra + ['--cfg-options'] + overrides
    if args.dry_run:
        import shlex
        print(shlex.join(command))
        return
    if not args.action.startswith('train') and not Path(args.checkpoint).is_file():
        parser.error('Brak checkpointu. Uruchom: python scripts/download_checkpoints.py')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
