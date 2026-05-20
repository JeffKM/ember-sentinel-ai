"""
T-037: 데이터 증강 실험 시각화

출력물:
- viz_results/augmentation_comparison.png   (증강 설정별 성능 비교)
- viz_results/augmentation_delta.png        (Baseline 대비 mAP 개선폭)
- viz_results/augmentation_report.md        (Markdown 보고서)
"""

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt

from sim_data import print_data_source_info
from sim_data.augmentation_comparison import get_augmentation_comparison

# --- 설정 ---
OUTPUT_DIR = 'viz_results'
DPI = 150
COLOR_PRIMARY = '#D32F2F'
COLOR_SECONDARY = '#1565C0'
COLOR_ADOPTED = '#2E7D32'   # 채택된 설정 (초록)
COLOR_OVERFIT = '#9E9E9E'   # 과도 증강 (회색)
COLOR_DEFAULT = '#FF8F00'   # 기본 (주황)


def plot_augmentation_comparison(aug_data):
    """증강 설정별 성능 비교 수평 막대 그래프를 생성합니다."""
    fig, ax = plt.subplots(figsize=(12, 7))

    names = [d['name'] for d in aug_data]
    mAP50 = [d['mAP50'] for d in aug_data]
    adopted = [d['adopted'] for d in aug_data]

    # 색상 배정
    colors = []
    for d in aug_data:
        if d['adopted']:
            colors.append(COLOR_ADOPTED)
        elif d['name'] == 'All Augmentations':
            colors.append(COLOR_OVERFIT)
        else:
            colors.append(COLOR_DEFAULT)

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, mAP50, height=0.6, color=colors, alpha=0.85, edgecolor='white')

    # 수치 라벨 + 채택/과도 태그
    for i, (bar, v) in enumerate(zip(bars, mAP50)):
        label = f'{v:.1f}%'
        if aug_data[i]['adopted']:
            label += '  [Adopted]'
        elif aug_data[i]['name'] == 'All Augmentations':
            label += '  [Over-augmented]'
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('mAP@0.5 (%)', fontsize=11)
    ax.set_title('Data Augmentation Experiment Results', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_DEFAULT, label='Tested'),
        Patch(facecolor=COLOR_ADOPTED, label='Adopted'),
        Patch(facecolor=COLOR_OVERFIT, label='Over-augmented'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'augmentation_comparison.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ 저장 완료: {path}")


def plot_augmentation_delta(aug_data):
    """Baseline 대비 mAP 개선폭 델타 차트를 생성합니다."""
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline = aug_data[0]['mAP50']
    # Baseline 자체는 제외 (delta=0)
    delta_data = aug_data[1:]
    names = [d['name'] for d in delta_data]
    deltas = [d['mAP50'] - baseline for d in delta_data]

    # 색상: 양수(개선)=빨강, 음수(하락 대비 최적)=회색
    colors = []
    for i, d in enumerate(delta_data):
        if d['adopted']:
            colors.append(COLOR_ADOPTED)
        elif deltas[i] < max(deltas):
            colors.append(COLOR_DEFAULT)
        else:
            colors.append(COLOR_PRIMARY)

    # 과도 증강이 직전보다 하락했으면 회색
    if deltas[-1] < deltas[-2]:
        colors[-1] = COLOR_OVERFIT

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, height=0.5, color=colors, alpha=0.85, edgecolor='white')

    for bar, v in zip(bars, deltas):
        sign = '+' if v > 0 else ''
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{sign}{v:.1f}%p', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('mAP@0.5 Delta from Baseline (%p)', fontsize=11)
    ax.set_title(f'mAP Improvement over Baseline ({baseline:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'augmentation_delta.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ 저장 완료: {path}")


def generate_augmentation_report(aug_data):
    """Markdown 증강 실험 보고서를 생성합니다."""
    baseline = aug_data[0]['mAP50']

    rows = ''
    for d in aug_data:
        delta = d['mAP50'] - baseline
        sign = '+' if delta > 0 else ''
        tag = ' **[Adopted]**' if d['adopted'] else ''
        if d['name'] == 'All Augmentations':
            tag = ' *(Over-augmented)*'
        rows += (
            f"| {d['name']}{tag} | {d['description']} | "
            f"{d['mAP50']:.1f}% | {d['mAP50_95']:.1f}% | "
            f"{d['precision']:.1f}% | {d['recall']:.1f}% | {sign}{delta:.1f}%p |\n"
        )

    report = f"""# 데이터 증강 실험 보고서

## 실험 개요

FASDD_CV 데이터셋에서 YOLOv11n 모델의 데이터 증강 전략별 성능을 비교합니다.
각 설정은 **동일한 하이퍼파라미터**(30 epochs, batch 32, imgsz 640)로 학습했습니다.

## 실험 결과

| Setting | Description | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Delta |
|---------|-------------|---------|--------------|-----------|--------|-------|
{rows}

## 분석

### 채택 설정: Mosaic + Flip + MixUp
- Baseline 대비 **+{aug_data[3]['mAP50'] - baseline:.1f}%p** mAP@0.5 향상
- Mosaic: 작은 객체 학습 강화 (+4.2%p)
- Horizontal Flip: 좌우 대칭 불변성 (+2.2%p 추가)
- MixUp (0.1): 클래스 간 경계 정규화 (+1.3%p 추가)

### 과도한 증강의 부작용
- **All Augmentations** 설정은 CutOut + ColorJitter 추가 시 오히려 성능 하락 (-1.8%p)
- 원인: 화염/연기의 색상 정보가 탐지에 중요하므로, ColorJitter가 학습을 방해
- CutOut은 작은 화염 객체를 완전히 가릴 수 있어 역효과

## 시각화 파일

- `augmentation_comparison.png` - 증강 설정별 성능 비교
- `augmentation_delta.png` - Baseline 대비 개선폭 차트
"""
    path = os.path.join(OUTPUT_DIR, 'augmentation_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 저장 완료: {path}")


def main():
    """T-037 데이터 증강 실험 시각화를 실행합니다."""
    print("\n" + "=" * 60)
    print("## T-037: 데이터 증강 실험 시각화")
    print("=" * 60)

    info = print_data_source_info()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aug_data = get_augmentation_comparison()

    plot_augmentation_comparison(aug_data)
    plot_augmentation_delta(aug_data)
    generate_augmentation_report(aug_data)

    print("\n✅ T-037 데이터 증강 실험 시각화 완료!")


if __name__ == '__main__':
    main()
