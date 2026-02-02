import pybullet as p
import pybullet_data
import time
import random
import os
import math
import numpy as np
import trimesh

# ==========================================
# [설정 영역]
# ==========================================
BATCH_SIZE = 40
SMALL_BATCH_SIZE = 5     # [NEW] 목표 근처에서 조금씩 넣기 위함
MAX_TOTAL_ITEMS = 1000
BASE_SHAKE_INTENSITY = 5.0
TOP_K_FOR_HEIGHT = 15    # [NEW] 상위 15개 아이템의 평균 높이를 사용
# ==========================================

def set_rendering(enable):
    """PyBullet 렌더링 On/Off 제어"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if enable else 0)

def wait_for_settle(item_ids, velocity_threshold=0.05, max_steps=2000, vis_mode=0):
    """안정화 대기 함수"""
    if vis_mode >= 1: set_rendering(False)

    settled = False
    for i in range(max_steps):
        p.stepSimulation()
        if i % 20 == 0:
            is_moving = False
            for uid in item_ids:
                lin_vel, _ = p.getBaseVelocity(uid)
                if np.linalg.norm(lin_vel) > velocity_threshold:
                    is_moving = True
                    break 
            if not is_moving:
                settled = True
                break
    
    if vis_mode >= 1:
        if vis_mode == 2:
            set_rendering(True)
            time.sleep(0.01)
            set_rendering(False) 

def evaluate_packing_state(item_ids, half_w_m, goal_height_m, top_k=15):
    """
    [NEW] 적재 상태를 정밀하게 평가하는 함수
    Returns:
        effective_height (float): 상위 k개의 평균 높이 (이상치 제거)
        valid_count_below_goal (int): 목표 높이 아래에 있는 유효 아이템 수
    """
    if not item_ids: return 0.0, 0
    
    boundary = half_w_m * 1.1
    valid_z_list = []      # 벽 안에 있는 모든 아이템의 Z좌표
    valid_count_below = 0  # 목표 높이 아래에 있는 아이템 수
    
    for uid in item_ids:
        pos, _ = p.getBasePositionAndOrientation(uid)
        
        # 1. 벽 밖으로 나간 것 제외
        if abs(pos[0]) > boundary or abs(pos[1]) > boundary:
            continue
            
        z = pos[2]
        valid_z_list.append(z)
        
        # 2. [핵심] 중심점이 목표 높이 이하인 것만 카운트
        if z <= goal_height_m:
            valid_count_below += 1
            
    if not valid_z_list:
        return 0.0, 0

    # 3. [핵심] 상위 K개 평균 높이 계산 (Effective Height)
    # Z좌표 내림차순 정렬
    valid_z_list.sort(reverse=True)
    
    # 아이템이 K개보다 적으면 전체 평균, 많으면 상위 K개만 평균
    k = min(len(valid_z_list), top_k)
    top_z_values = valid_z_list[:k]
    effective_height = sum(top_z_values) / k
    
    return effective_height, valid_count_below

def run_fast_physics_packing(filename, container_size, goal_height, vis_mode=0):
    # 단위 변환 (mm -> m)
    W_mm, L_mm, H_mm = container_size
    W_m, L_m, H_m = W_mm*0.001, L_mm*0.001, H_mm*0.001
    GOAL_HEIGHT_m = goal_height * 0.001
    STL_FILENAME = filename

    # 1. PyBullet 초기화
    try: p.disconnect()
    except: pass
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=20)
    p.resetDebugVisualizerCamera(cameraDistance=max(W_m, H_m)*1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,H_m/3])

    if vis_mode == 1:
        print("   [Vis] Mode 1: 렌더링을 끕니다. (최종 결과만 표시)")
        set_rendering(False)
    elif vis_mode == 2:
        print("   [Vis] Mode 2: 적응형 렌더링 (중간 과정 생략)")

    # 2. 환경 생성 (바닥/벽)
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=0.8, restitution=0.1)

    half_w = W_m / 2
    wall_h = H_m
    t = 0.01 
    def create_wall(pos, size):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.6, 0.6, 0.6, 0.3])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        p.createMultiBody(0, col, vis, basePosition=pos)
    
    overlap = 0.002
    create_wall([0, half_w + t/2 - overlap, wall_h/2], [half_w + t + overlap, t/2, wall_h/2])
    create_wall([0, -(half_w + t/2 - overlap), wall_h/2], [half_w + t + overlap, t/2, wall_h/2])
    create_wall([half_w + t/2 - overlap, 0, wall_h/2], [t/2, half_w, wall_h/2])
    create_wall([-(half_w + t/2 - overlap), 0, wall_h/2], [t/2, half_w, wall_h/2])

    # 3. 아이템 로드
    try:
        mesh = trimesh.load(STL_FILENAME)
        scale_factor = [0.001]*3 if np.max(mesh.extents) > 1.0 else [1.0]*3
        col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=STL_FILENAME, meshScale=scale_factor)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=STL_FILENAME, meshScale=scale_factor)
    except Exception as e:
        print(f"[Error] STL 로드 실패: {e}")
        return

    item_ids = []
    current_eff_height_m = 0.0
    
    print(f"=== 물리 시뮬레이션 시작 (Goal: {goal_height}mm) ===")
    start_time = time.time()

    while len(item_ids) < MAX_TOTAL_ITEMS:
        # [상태 평가] Top-K 평균 높이 사용
        current_eff_height_m, valid_cnt = evaluate_packing_state(
            item_ids, half_w, GOAL_HEIGHT_m, top_k=TOP_K_FOR_HEIGHT
        )
        
        # 종료 조건
        if current_eff_height_m >= GOAL_HEIGHT_m:
            print(f"\n[성공] 목표 높이 도달 (Effective): {current_eff_height_m * 1000:.1f}mm")
            break
            
        # [NEW] 목표 높이에 근접하면(80% 이상) 투하량을 줄여서 정밀하게 제어
        is_near_goal = current_eff_height_m > (GOAL_HEIGHT_m * 0.8)
        current_batch = SMALL_BATCH_SIZE if is_near_goal else BATCH_SIZE
        
        # 투하 높이
        drop_z_start = max(current_eff_height_m, 0.05) + W_m * 0.5
        
        if vis_mode == 2: set_rendering(False)
        
        # A. 투하
        for _ in range(current_batch):
            if len(item_ids) >= MAX_TOTAL_ITEMS: break
            x = random.uniform(-half_w*0.7, half_w*0.7)
            y = random.uniform(-half_w*0.7, half_w*0.7)
            z = drop_z_start + random.uniform(0, 0.05)
            orn = p.getQuaternionFromEuler([random.uniform(0,3.14) for _ in range(3)])
            
            uid = p.createMultiBody(0.1, col_shape, vis_shape, [x,y,z], orn)
            p.changeDynamics(uid, -1, lateralFriction=0.6, restitution=0.2, ccdSweptSphereRadius=0.002)
            item_ids.append(uid)
            
        # B. 1차 안정화
        wait_for_settle(item_ids, velocity_threshold=0.05, max_steps=500, vis_mode=vis_mode)

        # C. 적응형 흔들기 (Adaptive Shake)
        progress = min(current_eff_height_m / GOAL_HEIGHT_m, 1.0)
        damping_factor = max(0.2, 1.0 - progress) 
        current_shake_force = BASE_SHAKE_INTENSITY * damping_factor
        
        # 목표 근처에서는 흔들기도 최소화
        if is_near_goal: current_shake_force *= 0.5

        for t in range(20):
            fx = (random.random() - 0.5) * current_shake_force
            fy = (random.random() - 0.5) * current_shake_force
            for uid in item_ids:
                 p.applyExternalForce(uid, -1, [fx, fy, -2.0], [0,0,0], p.WORLD_FRAME)
            p.stepSimulation()
            if vis_mode == 0: time.sleep(0.001)

        # D. 최종 안정화
        wait_for_settle(item_ids, velocity_threshold=0.005, max_steps=800, vis_mode=vis_mode)

        if vis_mode == 2:
            set_rendering(True)
            p.stepSimulation()

        # 상태 재평가 및 로그
        current_eff_height_m, valid_cnt = evaluate_packing_state(
            item_ids, half_w, GOAL_HEIGHT_m, top_k=TOP_K_FOR_HEIGHT
        )
        print(f"   [진행] 투입:{len(item_ids)} | 유효:{valid_cnt} | 높이(Eff):{current_eff_height_m*1000:.1f}mm")

    if vis_mode >= 1: set_rendering(True)
        
    print(f"\n>>> 시뮬레이션 종료. (총 시간: {time.time()-start_time:.1f}초)")
    print(f">>> 최종 유효 적재 개수 (목표 높이 이하): {valid_cnt}개")
    
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    run_fast_physics_packing("llamp_mod.stl", [300, 300, 400], 115.0, vis_mode=0)
