"""
모델 비교 시뮬레이션 데이터

비교 축:
1. YOLOv11n(2.6M) vs YOLOv11s(9.4M)
2. epoch별 (10/20/30)
3. imgsz별 (320/480/640)
"""


def get_model_comparison():
    """
    YOLOv11n vs YOLOv11s 비교 데이터를 반환합니다.

    Returns:
        list[dict]: 모델별 성능 지표
    """
    return [
        {
            'model': 'YOLOv11n',
            'params_m': 2.6,
            'flops_g': 6.5,
            'model_size_mb': 5.2,
            'mAP50': 87.1,
            'mAP50_95': 54.5,
            'precision': 86.4,
            'recall': 83.8,
            'inference_ms': 78.5,    # RPi5 기준
            'fps': 12.7,
        },
        {
            'model': 'YOLOv11s',
            'params_m': 9.4,
            'flops_g': 21.5,
            'model_size_mb': 18.8,
            'mAP50': 89.3,
            'mAP50_95': 58.2,
            'precision': 88.1,
            'recall': 85.6,
            'inference_ms': 180.0,   # RPi5 기준
            'fps': 5.5,
        },
    ]


def get_epoch_comparison():
    """
    epoch별(10/20/30) 학습 성능 비교 데이터를 반환합니다.

    Returns:
        list[dict]: epoch별 성능 지표 (YOLOv11n 기준)
    """
    return [
        {
            'epochs': 10,
            'mAP50': 68.3,
            'mAP50_95': 38.7,
            'precision': 72.1,
            'recall': 65.4,
            'train_time_min': 45,
        },
        {
            'epochs': 20,
            'mAP50': 82.5,
            'mAP50_95': 49.8,
            'precision': 82.3,
            'recall': 79.1,
            'train_time_min': 90,
        },
        {
            'epochs': 30,
            'mAP50': 87.1,
            'mAP50_95': 54.5,
            'precision': 86.4,
            'recall': 83.8,
            'train_time_min': 135,
        },
    ]


def get_imgsz_comparison():
    """
    입력 이미지 크기별(320/480/640) 성능-속도 트레이드오프 데이터를 반환합니다.

    Returns:
        list[dict]: imgsz별 성능 지표 (YOLOv11n, RPi5 기준)
    """
    return [
        {
            'imgsz': 320,
            'mAP50': 78.2,
            'mAP50_95': 44.1,
            'inference_ms': 32.5,
            'fps': 30.8,
        },
        {
            'imgsz': 480,
            'mAP50': 84.0,
            'mAP50_95': 50.3,
            'inference_ms': 52.8,
            'fps': 18.9,
        },
        {
            'imgsz': 640,
            'mAP50': 87.1,
            'mAP50_95': 54.5,
            'inference_ms': 78.5,
            'fps': 12.7,
        },
    ]
