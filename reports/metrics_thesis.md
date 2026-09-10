# BEVFusion nuScenes validation metrics

- Checkpoint: `work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13/iter_741480.pth`
- Evaluation split: nuScenes validation (6019 samples)
- Protocol: `detection_cvpr_2019` (official nuScenes detection)
- Modalities: LiDAR + six surround-view cameras
- Authoritative source: `work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13/test_iter_741480/nuscenes_results/pred_instances_3d/metrics_summary.json`

## Global metrics

| NDS ↑ | mAP ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6769 | 0.6239 | 0.2849 | 0.2580 | 0.3329 | 0.2928 | 0.1816 |

## Per-class metrics

| Class | mean AP ↑ | AP@0.5 m ↑ | AP@1.0 m ↑ | AP@2.0 m ↑ | AP@4.0 m ↑ | trans. err ↓ | scale err ↓ | orient. err ↓ | vel. err ↓ | attr. err ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 0.8730 | 0.7866 | 0.8779 | 0.9081 | 0.9195 | 0.1729 | 0.1552 | 0.1026 | 0.3169 | 0.1874 |
| truck | 0.5952 | 0.4163 | 0.5962 | 0.6655 | 0.7028 | 0.3093 | 0.1822 | 0.1118 | 0.2728 | 0.2297 |
| bus | 0.7099 | 0.4754 | 0.6787 | 0.8303 | 0.8551 | 0.3355 | 0.1827 | 0.0694 | 0.5288 | 0.2506 |
| trailer | 0.4077 | 0.1249 | 0.3574 | 0.5270 | 0.6215 | 0.5246 | 0.2090 | 0.6130 | 0.2141 | 0.1437 |
| construction_vehicle | 0.2613 | 0.0358 | 0.2020 | 0.3536 | 0.4540 | 0.6741 | 0.4483 | 0.9440 | 0.1304 | 0.3074 |
| pedestrian | 0.8593 | 0.8408 | 0.8550 | 0.8650 | 0.8765 | 0.1332 | 0.2895 | 0.3874 | 0.2393 | 0.0899 |
| motorcycle | 0.5792 | 0.5108 | 0.5855 | 0.6049 | 0.6157 | 0.2055 | 0.2496 | 0.3411 | 0.4120 | 0.2345 |
| bicycle | 0.5168 | 0.4988 | 0.5191 | 0.5214 | 0.5279 | 0.1601 | 0.2583 | 0.3654 | 0.2279 | 0.0095 |
| traffic_cone | 0.7603 | 0.7349 | 0.7460 | 0.7656 | 0.7947 | 0.1330 | 0.3196 | N/A | N/A | N/A |
| barrier | 0.6765 | 0.5779 | 0.6752 | 0.7193 | 0.7338 | 0.2007 | 0.2861 | 0.0611 | N/A | N/A |

## Runtime

- Final-window data time: 0.0032 s/sample
- Final-window test-loop time: 0.3636 s/sample
- Approximate throughput: 2.7503 samples/s

Runtime is the final MMEngine logger-window measurement and includes model inference plus loop overhead; it is not a hardware-independent model latency benchmark.

## Metric interpretation

- NDS and mAP: higher is better.
- mATE (m), mASE (1 - scale IoU), mAOE (rad), mAVE (m/s), and mAAE (1 - attribute accuracy): lower is better.
- AP is averaged over the four center-distance thresholds for the reported mAP.
- `N/A` denotes a nuScenes metric that is not defined for that class.
