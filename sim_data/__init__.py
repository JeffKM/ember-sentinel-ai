"""
sim_data 패키지 - 시뮬레이션/실제 학습 데이터 자동 감지 모듈

실제 학습 결과(experiments/)가 존재하면 실제 데이터를 반환하고,
없으면 ADR-002 기반 시뮬레이션 데이터를 반환합니다.
"""

import os

# 실제 학습 결과 경로
EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')
REAL_RESULTS_CSV = os.path.join(EXPERIMENTS_DIR, 'yolov11n', 'results.csv')


def get_data_source():
    """
    데이터 소스를 자동 감지합니다.

    Returns:
        dict: {'source': 'real' | 'simulation', 'path': str | None}
    """
    if os.path.exists(REAL_RESULTS_CSV):
        return {
            'source': 'real',
            'path': REAL_RESULTS_CSV,
        }
    return {
        'source': 'simulation',
        'path': None,
    }


def print_data_source_info():
    """현재 데이터 소스 정보를 출력합니다."""
    info = get_data_source()
    if info['source'] == 'real':
        print(f"📊 데이터 소스: 실제 학습 결과 ({info['path']})")
    else:
        print("📊 데이터 소스: 시뮬레이션 (ADR-002 기반 수치)")
    return info
