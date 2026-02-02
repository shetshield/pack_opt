import argparse
import os
import sys

# 모듈 로딩 시도
try:
    import analyze_object
    import mathematical_optimization
    import pack_physics_sim
    import voxel_optimization
except ImportError as e:
    print(f"[Critical Error] 필수 모듈을 찾을 수 없습니다: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Smart Auto-Packing Simulator (Unit: mm)")
    
    parser.add_argument("filename", type=str, help="STL 파일 경로")
    parser.add_argument("--container", type=float, nargs=3, default=[300, 300, 400], help="컨테이너 W L H (mm)")
    parser.add_argument("--goal", type=float, default=115.0, help="목표 높이 (mm)")
    parser.add_argument("--mode", type=str, choices=['auto', 'math', 'physics', 'voxel'], default='auto', help="실행 모드")
    
    # [NEW] 시각화 모드 설정
    # 0: Full (Default), 1: Final Only (Fast), 2: Adaptive (Balanced)
    parser.add_argument("--vis", type=int, choices=[0, 1, 2], default=0, help="시각화 모드 (0:Full, 1:ResultOnly, 2:Adaptive)")

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"[Error] 파일 '{args.filename}'을 찾을 수 없습니다.")
        return

    print(f"\n>>> [1/3] Object Analysis: {args.filename}")
    analyzer = analyze_object.ShapeAnalyzer(args.filename)
    suggested_mode, item_dims = analyzer.analyze()
    
    final_mode = suggested_mode
    if args.mode != 'auto':
        final_mode = "MATH_OPTIMIZER" if args.mode == 'math' else "PHYSICS_SIM" if args.mode == 'physics' else "VOXEL_PACKER"
        print(f"  [Override] '{args.mode.upper()}' 모드 실행")
    else:
        print(f"  [Auto] '{final_mode}' 모드 선택됨")

    print(f"\n>>> [2/3] Simulation Start (Goal: {args.goal}mm, Vis: {args.vis})")
    
    if final_mode == "MATH_OPTIMIZER":
        optimizer = mathematical_optimization.MixedLayerOptimizer(args.container)
        optimizer.optimize_mixed_stacking(item_dims, args.goal)

    elif final_mode == "PHYSICS_SIM":
        pack_physics_sim.run_fast_physics_packing(
            filename=args.filename,
            container_size=args.container,
            goal_height=args.goal,
            vis_mode=args.vis # [NEW] 시각화 모드 전달
        )

    elif final_mode == "VOXEL_PACKER":
        voxel_optimization.run_voxel_packing(
            filename=args.filename,
            container_size=args.container,
            goal_height=args.goal
        )
    
    print("\n>>> [3/3] Completed.")

if __name__ == "__main__":
    main()