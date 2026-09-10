# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import ast
import inspect
import os
import os.path as osp

import torch

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


def register_local_checkpoint_compat():
    """Restore trusted local checkpoint loading under PyTorch 2.6+."""
    try:
        load_params = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        load_params = {}

    if 'weights_only' not in load_params:
        return

    from mmengine.runner.checkpoint import CheckpointLoader

    def load_from_local(filename, map_location):
        filename = osp.expanduser(filename)
        if not osp.isfile(filename):
            raise FileNotFoundError(f'{filename} can not be found.')
        return torch.load(filename, map_location=map_location, weights_only=False)

    CheckpointLoader.register_scheme(prefixes='', loader=load_from_local, force=True)


def parse_indices(value: str):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return int(value)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    if isinstance(parsed, int):
        return parsed
    raise ValueError(f'Unsupported indices value: {value}')


def format_box(box):
    center = f'({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})'
    size = f'({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f})'
    yaw = f'{box[6]:.2f}' if len(box) > 6 else 'n/a'
    parts = [f'center={center}', f'size={size}', f'yaw={yaw}']
    if len(box) > 8:
        parts.append(f'vel=({box[7]:.2f}, {box[8]:.2f})')
    return '  '.join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Print LiDAR 3D detections in the terminal.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='work dir for runner logs')
    parser.add_argument('--score-thr', type=float, default=0.35,
                        help='minimum score to print')
    parser.add_argument('--topk', type=int, default=20,
                        help='maximum detections to print per frame')
    parser.add_argument('--indices', type=str,
                        help='subset of test samples, e.g. 8 or "[0,10,25]"')
    parser.add_argument('--ceph', action='store_true',
                        help='Use ceph as data storage backend')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override config options')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    register_local_checkpoint_compat()

    cfg = Config.fromfile(args.config)
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    if args.indices is not None:
        cfg.test_dataloader.dataset.indices = parse_indices(args.indices)

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.load_checkpoint(args.checkpoint)
    runner.model.eval()

    classes = runner.test_dataloader.dataset.metainfo.get('classes', ())

    frame_idx = 0
    for data_batch in runner.test_dataloader:
        with torch.no_grad():
            outputs = runner.model.test_step(data_batch)

        for data_sample in outputs:
            sample_idx = getattr(data_sample, 'sample_idx', 'n/a')
            lidar_path = getattr(data_sample, 'lidar_path', 'n/a')
            print(f'Frame {frame_idx} | sample_idx={sample_idx} | lidar={lidar_path}')

            pred = getattr(data_sample, 'pred_instances_3d', None)
            if pred is None or len(pred) == 0:
                print('  <no detections>')
                frame_idx += 1
                continue

            scores = pred.scores_3d.detach().cpu().numpy()
            labels = pred.labels_3d.detach().cpu().numpy()
            boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
            order = scores.argsort()[::-1]

            printed = 0
            for det_idx in order:
                score = float(scores[det_idx])
                if score < args.score_thr:
                    continue
                label_id = int(labels[det_idx])
                label_name = classes[label_id] if label_id < len(classes) else str(label_id)
                box_desc = format_box(boxes[det_idx])
                print(f'  - {label_name:<22} score={score:.3f}  {box_desc}')
                printed += 1
                if printed >= args.topk:
                    break

            if printed == 0:
                print(f'  <no detections above score_thr={args.score_thr}>')
            frame_idx += 1


if __name__ == '__main__':
    main()
