"""
디바이스별 추론 벤치마크 시뮬레이션 데이터

대상 디바이스:
- Raspberry Pi 5 (주력 엣지 디바이스)
- MacBook M2 (개발 환경)
- GPU Server (RTX 3060, 학습/검증 환경)
- Raspberry Pi 4 (구형 참조)
"""


def get_benchmark_data():
    """
    디바이스별 추론 속도 벤치마크 데이터를 반환합니다.

    시간 단위: 밀리초(ms), YOLOv11n + NCNN FP16 기준

    Returns:
        list[dict]: 디바이스별 벤치마크 지표
    """
    return [
        {
            'device': 'Raspberry Pi 5',
            'chip': 'BCM2712 (Cortex-A76)',
            'ram': '8GB LPDDR4X',
            'accelerator': 'ARM NEON',
            'preprocess_ms': 8.2,
            'inference_ms': 62.5,
            'postprocess_ms': 7.8,
            'total_ms': 78.5,
            'fps': 12.7,
            'power_w': 5.0,
        },
        {
            'device': 'MacBook M2',
            'chip': 'Apple M2',
            'ram': '16GB Unified',
            'accelerator': 'Apple Neural Engine',
            'preprocess_ms': 1.8,
            'inference_ms': 8.2,
            'postprocess_ms': 2.3,
            'total_ms': 12.3,
            'fps': 81.3,
            'power_w': 15.0,
        },
        {
            'device': 'GPU Server',
            'chip': 'Intel i7-12700K',
            'ram': '32GB DDR5',
            'accelerator': 'NVIDIA RTX 3060',
            'preprocess_ms': 0.8,
            'inference_ms': 2.8,
            'postprocess_ms': 1.2,
            'total_ms': 4.8,
            'fps': 208.3,
            'power_w': 170.0,
        },
        {
            'device': 'Raspberry Pi 4',
            'chip': 'BCM2711 (Cortex-A72)',
            'ram': '4GB LPDDR4',
            'accelerator': 'ARM NEON',
            'preprocess_ms': 18.5,
            'inference_ms': 145.0,
            'postprocess_ms': 21.5,
            'total_ms': 185.0,
            'fps': 5.4,
            'power_w': 6.0,
        },
    ]
