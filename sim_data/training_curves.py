"""
학습 곡선, 혼동 행렬, PR 곡선 시뮬레이션 데이터

ADR-002 기반 핵심 수치:
- mAP@0.5: 87.1%, mAP@0.5:0.95: 54.5%
- fire AP: 90.3%, smoke AP: 83.9%
- 30 epoch 학습
"""

import numpy as np


def generate_training_curves(epochs=30, seed=42):
    """
    30 epoch 학습 곡선 데이터를 생성합니다.

    Returns:
        dict: epoch, box_loss, cls_loss, dfl_loss, val_box_loss, val_cls_loss,
              val_dfl_loss, mAP50, mAP50_95, precision, recall, lr
    """
    rng = np.random.RandomState(seed)
    ep = np.arange(1, epochs + 1)

    # --- 학습 Loss 곡선 (지수 감소 + 노이즈) ---
    # Box loss: 1.8 → 0.45
    box_loss = 1.8 * np.exp(-0.12 * ep) + 0.45 + rng.normal(0, 0.02, epochs)
    # Cls loss: 2.5 → 0.60
    cls_loss = 2.5 * np.exp(-0.14 * ep) + 0.60 + rng.normal(0, 0.03, epochs)
    # DFL loss: 1.4 → 0.95
    dfl_loss = 1.4 * np.exp(-0.10 * ep) + 0.95 + rng.normal(0, 0.015, epochs)

    # --- 검증 Loss (학습보다 살짝 높음) ---
    val_box_loss = box_loss + 0.08 + rng.normal(0, 0.01, epochs)
    val_cls_loss = cls_loss + 0.12 + rng.normal(0, 0.015, epochs)
    val_dfl_loss = dfl_loss + 0.05 + rng.normal(0, 0.01, epochs)

    # --- mAP 곡선 (시그모이드 형태) ---
    # mAP@0.5: 최종 87.1%
    mAP50 = 87.1 / (1 + np.exp(-0.25 * (ep - 10))) + rng.normal(0, 0.5, epochs)
    mAP50 = np.clip(mAP50, 0, 87.5)
    mAP50[-1] = 87.1  # 최종값 고정

    # mAP@0.5:0.95: 최종 54.5%
    mAP50_95 = 54.5 / (1 + np.exp(-0.22 * (ep - 12))) + rng.normal(0, 0.4, epochs)
    mAP50_95 = np.clip(mAP50_95, 0, 55.0)
    mAP50_95[-1] = 54.5

    # --- Precision / Recall ---
    precision = 85.0 / (1 + np.exp(-0.28 * (ep - 8))) + rng.normal(0, 0.6, epochs)
    precision = np.clip(precision, 0, 90.0)
    precision[-1] = 86.4

    recall = 82.0 / (1 + np.exp(-0.24 * (ep - 9))) + rng.normal(0, 0.5, epochs)
    recall = np.clip(recall, 0, 86.0)
    recall[-1] = 83.8

    # --- Learning Rate (cosine decay) ---
    lr0 = 0.01
    lr = lr0 * 0.5 * (1 + np.cos(np.pi * ep / epochs))

    return {
        'epoch': ep,
        'box_loss': box_loss,
        'cls_loss': cls_loss,
        'dfl_loss': dfl_loss,
        'val_box_loss': val_box_loss,
        'val_cls_loss': val_cls_loss,
        'val_dfl_loss': val_dfl_loss,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'precision': precision,
        'recall': recall,
        'lr': lr,
    }


def generate_confusion_matrix():
    """
    혼동 행렬 데이터를 생성합니다. (fire / smoke / background)

    Returns:
        dict: matrix (3x3 ndarray), labels (list)
    """
    # 정규화된 혼동 행렬 (행: 실제, 열: 예측)
    # fire: 높은 정밀도, smoke: 약간 혼동, background: 소수 오탐
    matrix = np.array([
        [0.903, 0.042, 0.055],   # fire → fire/smoke/bg
        [0.051, 0.839, 0.110],   # smoke → fire/smoke/bg
        [0.030, 0.045, 0.925],   # bg → fire/smoke/bg
    ])
    labels = ['fire', 'smoke', 'background']

    return {'matrix': matrix, 'labels': labels}


def generate_pr_curves(num_points=100, seed=42):
    """
    클래스별 PR 곡선 데이터를 생성합니다.

    Returns:
        dict: fire (recall, precision, ap), smoke (recall, precision, ap)
    """
    rng = np.random.RandomState(seed)
    recall_pts = np.linspace(0, 1, num_points)

    # fire PR 곡선 (AP = 90.3%)
    # np.maximum으로 음수 방지 (fractional power 경고 회피)
    fire_drop = np.maximum(recall_pts - 0.85, 0)
    fire_precision = np.where(
        recall_pts < 0.85,
        0.98 - 0.08 * recall_pts + rng.normal(0, 0.005, num_points),
        0.98 - 0.08 * 0.85 - 2.5 * fire_drop ** 1.5
    )
    fire_precision = np.clip(fire_precision, 0, 1)

    # smoke PR 곡선 (AP = 83.9%)
    smoke_drop = np.maximum(recall_pts - 0.75, 0)
    smoke_precision = np.where(
        recall_pts < 0.75,
        0.95 - 0.12 * recall_pts + rng.normal(0, 0.006, num_points),
        0.95 - 0.12 * 0.75 - 2.0 * smoke_drop ** 1.3
    )
    smoke_precision = np.clip(smoke_precision, 0, 1)

    return {
        'recall': recall_pts,
        'fire': {'precision': fire_precision, 'ap': 90.3},
        'smoke': {'precision': smoke_precision, 'ap': 83.9},
    }
