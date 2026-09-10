# BEVFusion — nuScenes

Prywatny zapis projektu MMDetection3D: detekcja 3D z LiDAR-u i sześciu kamer,
trening LiDAR, dostrajanie fuzji, walidacja i test. Kod modelu znajduje się
w `projects/BEVFusion/bevfusion/`, konfiguracje w `projects/BEVFusion/configs/`.
Zachowano lokalne poprawki obsługi checkpointów PyTorch i kompilacji dla Blackwell.
Pochodzenie kodu: `artifacts/source.json`; licencja: `LICENSE` (Apache-2.0).
Dokumentacja źródłowa: [README.upstream.md](README.upstream.md).

## Instalacja

Linux, Python 3.10, karta NVIDIA i CUDA Toolkit z `nvcc`.
Oryginalne środowisko: PyTorch 2.7.1+cu128, CUDA Toolkit 12.9, MMCV 2.1.0,
MMEngine 0.10.7, MMDetection 3.2.0. Skrypt domyślnie buduje dla Blackwell SM 12.0.

```bash
conda create -n bevfusion_blackwell python=3.10 -y
conda activate bevfusion_blackwell
# CUDA Toolkit 12.9 musi być zainstalowany osobno.
export CUDA_HOME=/usr/local/cuda-12.9  # dostosuj do swojej instalacji
bash scripts/install.sh
```

`requirements.txt` zawiera zależności upstream, `requirements/bevfusion.txt`
zależności eksperymentu, a `requirements/environment-observed.txt` pełny
informacyjny spis pakietów zastanego środowiska (także niezwiązanych z modelem).
Czysta instalacja od zera nie została zweryfikowana; rozszerzenia CUDA wymagają
kompilacji na docelowej maszynie. Nie instaluj całego spisu informacyjnego zamiast
uruchomienia skryptu instalacyjnego.

## Wagi modelu

W prywatnym wydaniu `v1.0.0` są trzy pliki: końcowy checkpoint fuzji
`iter_741480.pth`, startowy LiDAR `epoch_13.pth` oraz pretraining kamery
`swint-nuimages-pretrained.pth`. Zawierają rzeczywiste lokalne wagi, a nie
wyłącznie definicję architektury. Pobieranie wymaga dostępu do repozytorium:

```bash
gh auth login
python scripts/download_checkpoints.py
```

Skrypt sprawdza rozmiary i SHA-256 z `artifacts/checkpoints.json` i umieszcza
pliki w katalogach oczekiwanych przez konfigurację. Checkpointy zawierają
również stan treningu; loader zachowuje lokalną obsługę `weights_only=False`.

## Dane

Umieść rozpakowane nuScenes w `data/nuscenes/` (może być dowiązanie):
`samples/`, `sweeps/`, `maps/`, `v1.0-trainval/` i dla testu `v1.0-test/`.
Dane nie są dołączone do repozytorium. Standardowy konwerter przygotowuje
zarówno trainval, jak i test, więc poniższe polecenie wymaga obu części:

```bash
python tools/create_data.py nuscenes --root-path data/nuscenes \
  --out-dir data/nuscenes --extra-tag nuscenes --version v1.0
```

Wymagane pliki to `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`,
`nuscenes_infos_test.pkl` oraz baza obiektów dla treningu LiDAR.

## Trening, walidacja i test

Uruchamiaj polecenia w aktywnym środowisku po instalacji rozszerzeń.

```bash
# LiDAR od podstaw, 16 epok (batch size 5; dostosuj do pamięci GPU)
python scripts/run.py train-lidar
# Fuzja: 6 epok, batch size 1, start z dołączonego LiDAR epoch_13 + Swin
python scripts/run.py train
# Wznowienie przerwanego treningu fuzji
python scripts/run.py train --resume auto
# Oficjalne metryki na zbiorze walidacyjnym (6019 próbek)
python scripts/run.py validate
# Predykcje na walidacji bez liczenia metryk
python scripts/run.py predict
# Predykcje na osobnym zbiorze nuScenes test, eksport do oceny serwerowej
python scripts/run.py test
# Własne wagi / dodatkowe parametry MMEngine
python scripts/run.py validate --checkpoint /path/model.pth
python scripts/run.py train --cfg-options train_dataloader.batch_size=1
```

`test` korzysta z `nuscenes_infos_test.pkl` i `format_only=True`: etykiety
oficjalnego testu nie są lokalnie dostępne. `validate` liczy mAP/NDS.
Oba polecenia wykorzystują upstream `tools/test.py` z odpowiednią konfiguracją.
Wyniki trafiają do `work_dirs/<nazwa-modelu>/<validate|test|predict>/`.
Opcja `--dry-run` wyświetla komendę bez uruchamiania obliczeń.

Dostępny jest też `bash tools/test_bevfusion_iter_741480.sh` (walidacja wraz
z tabelami metryk, wymaga środowiska Conda o nazwie `bevfusion_blackwell`),
a `tools/analysis_tools/summarize_nuscenes_metrics.py` tworzy raporty CSV/Markdown.
Pełne narzędzia upstream do inferencji i wizualizacji są w `demo/` i `tools/`.

## Zapisane wyniki i weryfikacja

Istniejąca walidacja `iter_741480.pth`: **NDS 0.6769, mAP 0.6239**.
Raporty i źródłowy `metrics_summary.json` znajdują się w `reports/`.
Są to wyniki wcześniejszego przebiegu, nie nowej walidacji wykonanej przy eksporcie.
Trening i pełna ewaluacja nie uruchamiają się podczas tworzenia repozytorium.

Testy biblioteki są w `tests/`; wymagają odpowiednich zależności, część także GPU.
Podstawowe sprawdzenie składni i konfiguracji poleceń:

```bash
python -m compileall -q scripts projects/BEVFusion tools/train.py tools/test.py
python scripts/run.py train --dry-run
python scripts/run.py validate --dry-run
python scripts/run.py test --dry-run
```
