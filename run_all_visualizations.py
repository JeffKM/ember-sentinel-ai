"""
전체 시각화 일괄 실행 스크립트

모든 시각화를 순서대로 실행하고 결과를 요약합니다.
"""

import os
import time

from sim_data import print_data_source_info


def main():
    """전체 시각화를 일괄 실행합니다."""
    print("=" * 60)
    print("  Ember Sentinel AI - 전체 시각화 실행")
    print("=" * 60)

    info = print_data_source_info()
    start = time.time()

    # --- T-035: 학습 결과 시각화 ---
    from visualize_training import main as run_training
    run_training()

    # --- T-036: 모델 비교 실험 ---
    from visualize_comparison import main as run_comparison
    run_comparison()

    # --- T-037: 데이터 증강 실험 ---
    from visualize_augmentation import main as run_augmentation
    run_augmentation()

    # --- T-038: 추론 속도 벤치마크 ---
    from visualize_benchmark import main as run_benchmark
    run_benchmark()

    elapsed = time.time() - start

    # 결과 요약
    output_dir = 'viz_results'
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        png_files = [f for f in files if f.endswith('.png')]
        md_files = [f for f in files if f.endswith('.md')]

        print("\n" + "=" * 60)
        print("  전체 시각화 완료!")
        print("=" * 60)
        print(f"  소요 시간: {elapsed:.1f}초")
        print(f"  출력 디렉토리: {output_dir}/")
        print(f"  PNG 차트: {len(png_files)}개")
        for f in png_files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"    - {f} ({size / 1024:.1f}KB)")
        print(f"  MD 보고서: {len(md_files)}개")
        for f in md_files:
            print(f"    - {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
