# 🧠 Ember Sentinel AI

> **YOLOv11n 기반 화재·연기 감지 모델 학습 및 엣지 배포 파이프라인**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11n-Ultralytics-FF6F00)](https://docs.ultralytics.com/)
[![NCNN](https://img.shields.io/badge/NCNN-FP16-00599C)](https://github.com/Tencent/ncnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

인하대학교 캡스톤 디자인 프로젝트(팀: `inha-capstone-04`) — [Ember Sentinel](https://github.com/JeffKM/ember-sentinel) 시스템의 AI 모델 학습 파이프라인입니다. FASDD_CV 데이터셋으로 YOLOv11n 모델을 학습하고, Raspberry Pi 5 배포를 위해 NCNN 포맷으로 변환합니다.

---

## 목차

- [모델 성능](#모델-성능)
- [ML 파이프라인](#ml-파이프라인)
- [시작하기](#시작하기)
- [학습 설정](#학습-설정)
- [시각화](#시각화)
- [프로젝트 구조](#프로젝트-구조)
- [관련 레포지토리](#관련-레포지토리)

---

## 모델 성능

### 최종 성능 (YOLOv11n, FASDD_CV 테스트셋)

| 지표 | 수치 |
|------|------|
| mAP@0.5 | **87.1%** |
| mAP@0.5:0.95 | 54.5% |
| Precision | 86.4% |
| Recall | 83.8% |
| Fire AP | **90.3%** |
| Smoke AP | **83.9%** |

### 엣지 추론 속도

| 디바이스 | 칩셋 | 추론 시간 | FPS | 실시간 |
|----------|-------|-----------|-----|--------|
| **Raspberry Pi 5** | BCM2712 Cortex-A76 | **78.5ms** | 12.7 | ✓ |
| MacBook M2 | Apple M2 | 12.3ms | 81.3 | ✓ |
| GPU 서버 | i7-12700K + RTX3060 | 4.8ms | 208.3 | ✓ |
| Raspberry Pi 4 | BCM2711 Cortex-A72 | 185.0ms | 5.4 | ✗ |

> 실시간 기준: 100ms 이하 (10 FPS 이상)

### 모델 사양

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv11n (Nano) |
| 파라미터 | 2.6M |
| FLOPs | 6.5G |
| 모델 크기 | 5.2MB (NCNN FP16) |
| 입력 크기 | 640×640 |
| 감지 클래스 | `fire` (0), `smoke` (1) |

---

## ML 파이프라인

```
FASDD_CV 데이터셋 → 전처리 → EDA → 학습 → NCNN 변환 → 평가 → 시각화
```

| 단계 | 스크립트 | 설명 |
|------|---------|------|
| 1. 전처리 | `preprocessing.py` | FASDD_CV → YOLO 표준 디렉토리 구조 변환 |
| 2. EDA | `eda.py` | 클래스 분포, 바운딩박스 면적/종횡비 분석 |
| 3. 학습 | `train.py` | YOLOv11n 파인튜닝 (30 epochs, AMP, Early Stopping) |
| 4. 변환 | `export.py` | PyTorch → NCNN (FP16 Half Precision) |
| 5. 평가 | `inference.py` | NCNN 모델 테스트셋 mAP/Precision/Recall 측정 |
| 6. 시각화 | `visualize_*.py` | 학습 곡선, 모델 비교, 벤치마크 차트 생성 |

---

## 시작하기

### 사전 요구사항

- **Python** 3.11+
- **GPU** (학습 시 권장, CUDA 지원)

### 환경 설정

```bash
git clone https://github.com/JeffKM/ember-sentinel-ai.git
cd ember-sentinel-ai
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### 데이터셋 준비

FASDD_CV 데이터셋을 [Google Drive](https://drive.google.com/file/d/1TdJvs7Q0_ylIxBQvLI_U8plnoSItS-SX/view?usp=sharing)에서 다운로드 후 프로젝트 루트에 배치:

```bash
# 다운로드 후
unzip ./FASDD_CV.zip
python preprocessing.py
```

### 학습 실행

```bash
# 데이터 분석 (선택)
python eda.py

# 모델 학습 (GPU 권장)
python train.py

# NCNN 변환
python export.py

# 테스트셋 평가
python inference.py
```

### 사전 학습된 모델 사용

학습된 모델을 [Google Drive](https://drive.google.com/file/d/1RBWNbcXIIVwbMlQNk9-QOH2hBbzR9vog/view?usp=sharing)에서 다운로드:

```bash
# experiments/ 폴더에 배치
unzip ./experiments/yolov11n.zip -d ./experiments/
```

---

## 학습 설정

### 하이퍼파라미터

```python
model     = "yolo11n.pt"     # 사전 학습 가중치
epochs    = 30               # 학습 에포크
batch     = 32               # 배치 크기
imgsz     = 640              # 입력 이미지 크기
workers   = 2                # 데이터 로더 워커
patience  = 5                # Early Stopping
amp       = True             # 자동 혼합 정밀도
compile   = True             # 그래프 최적화
cache     = True             # RAM 캐싱
```

### 데이터 증강 전략

| 증강 기법 | mAP@0.5 | 기준선 대비 | 채택 |
|-----------|---------|-------------|------|
| Baseline (없음) | 79.4% | - | - |
| Mosaic | 83.6% | +4.2%p | ✓ |
| Mosaic + Flip | 85.8% | +6.4%p | ✓ |
| **Mosaic + Flip + MixUp** | **87.1%** | **+7.7%p** | **✓ 채택** |
| 전체 증강 (과적합) | 85.3% | +5.9%p | ✗ |

> CutOut, ColorJitter 추가 시 오히려 성능 하락 — 화재/연기의 색상 정보가 감지에 중요하기 때문

### 모델 선택 근거 (YOLOv11n vs YOLOv11s)

| 항목 | YOLOv11n (채택) | YOLOv11s |
|------|----------------|----------|
| 파라미터 | 2.6M | 9.4M |
| 모델 크기 | 5.2MB | 18.8MB |
| mAP@0.5 | 87.1% | 89.3% |
| RPi5 추론 | 78.5ms | 180.0ms |
| RPi5 FPS | 12.7 | 5.5 |
| 실시간 가능 | ✓ | ✗ |

> mAP 차이 2.2%p 대비 추론 속도 2.3배 빠름 → **Nano 선택**

---

## 시각화

모든 시각화를 한 번에 실행:

```bash
python run_all_visualizations.py
```

### 생성되는 차트 (viz_results/)

| 파일 | 설명 | 태스크 |
|------|------|--------|
| `training_dashboard.png` | Loss, mAP, Precision/Recall, LR 대시보드 | T-035 |
| `confusion_matrix.png` | 정규화된 혼동 행렬 (fire/smoke/background) | T-035 |
| `pr_curves.png` | 클래스별 Precision-Recall 곡선 | T-035 |
| `model_comparison.png` | YOLOv11n vs YOLOv11s 비교 | T-036 |
| `epoch_comparison.png` | 에포크별 성능 비교 (10/20/30) | T-036 |
| `imgsz_tradeoff.png` | 입력 크기 정확도-속도 트레이드오프 | T-036 |
| `augmentation_comparison.png` | 증강 전략별 성능 비교 | T-037 |
| `augmentation_delta.png` | 기준선 대비 개선폭 | T-037 |
| `benchmark_pipeline.png` | 디바이스별 추론 파이프라인 분석 | T-038 |
| `benchmark_fps.png` | 디바이스별 FPS 비교 | T-038 |

---

## 프로젝트 구조

```
ember-sentinel-ai/
├── preprocessing.py             # FASDD_CV 데이터셋 전처리
├── eda.py                       # 탐색적 데이터 분석
├── train.py                     # YOLOv11n 학습
├── export.py                    # PyTorch → NCNN 변환
├── inference.py                 # 테스트셋 평가
├── visualize_training.py        # 학습 결과 시각화 (T-035)
├── visualize_comparison.py      # 모델 비교 시각화 (T-036)
├── visualize_augmentation.py    # 증강 전략 시각화 (T-037)
├── visualize_benchmark.py       # 벤치마크 시각화 (T-038)
├── run_all_visualizations.py    # 전체 시각화 실행
├── sim_data/                    # 시뮬레이션 데이터 모듈
│   ├── __init__.py              # 실제/시뮬레이션 데이터 자동 감지
│   ├── training_curves.py       # 학습 곡선 데이터
│   ├── model_comparison.py      # 모델 비교 데이터
│   ├── augmentation_comparison.py # 증강 비교 데이터
│   └── benchmark_data.py        # 벤치마크 데이터
├── eda_results/                 # EDA 차트 출력
├── viz_results/                 # 시각화 차트 + 리포트 출력
├── requirements.txt             # Python 의존성
└── FASDD_CV/                    # 데이터셋 (다운로드 필요)
```

---

## 관련 레포지토리

| 레포지토리 | 역할 | 기술 스택 |
|------------|------|-----------|
| [ember-sentinel](https://github.com/JeffKM/ember-sentinel) | 모바일 앱 | React Native 0.81, Expo 54 |
| [ember-sentinel-server](https://github.com/JeffKM/ember-sentinel-server) | 백엔드 API | Java 17, Spring Boot 3.5 |
| [edge-IoT](https://github.com/JeffKM/edge-IoT) | 엣지 디바이스 | Python, OpenCV, LiveKit SDK |
| [Terraform-Bastion-Server](https://github.com/JeffKM/Terraform-Bastion-Server) | AWS 인프라 IaC | Terraform, AWS |

---

<div align="center">

**인하대학교 캡스톤 디자인 — 팀 `inha-capstone-04`**

</div>
