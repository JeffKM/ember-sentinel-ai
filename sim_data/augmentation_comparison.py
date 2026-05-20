"""
데이터 증강 실험 비교 시뮬레이션 데이터

5가지 증강 설정: Baseline → Mosaic → +Flip → +MixUp → All(과도)
"""


def get_augmentation_comparison():
    """
    증강 설정별 성능 비교 데이터를 반환합니다.

    'All' 설정은 과도한 증강으로 인해 성능이 하락하는 사례를 보여줍니다.

    Returns:
        list[dict]: 증강 설정별 성능 지표
    """
    return [
        {
            'name': 'Baseline',
            'description': 'No augmentation',
            'mAP50': 79.4,
            'mAP50_95': 46.2,
            'precision': 80.1,
            'recall': 76.3,
            'adopted': False,
        },
        {
            'name': 'Mosaic',
            'description': 'Mosaic only',
            'mAP50': 83.6,
            'mAP50_95': 50.1,
            'precision': 83.5,
            'recall': 80.2,
            'adopted': False,
        },
        {
            'name': 'Mosaic+Flip',
            'description': 'Mosaic + Horizontal Flip',
            'mAP50': 85.8,
            'mAP50_95': 52.7,
            'precision': 85.2,
            'recall': 82.5,
            'adopted': False,
        },
        {
            'name': 'Mosaic+Flip+MixUp',
            'description': 'Mosaic + Flip + MixUp (0.1)',
            'mAP50': 87.1,
            'mAP50_95': 54.5,
            'precision': 86.4,
            'recall': 83.8,
            'adopted': True,
        },
        {
            'name': 'All Augmentations',
            'description': 'Mosaic + Flip + MixUp + CutOut + ColorJitter',
            'mAP50': 85.3,
            'mAP50_95': 52.0,
            'precision': 84.0,
            'recall': 82.1,
            'adopted': False,
        },
    ]
