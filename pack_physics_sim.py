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
MAX_TOTAL_ITEMS = 1000
# 흔들기 강도 (기존 10.0 -> 5.0으로 하향 조정)
# m 단위계에서 5.0N은 0.1kg 물체에 5.0G 가속도를 줍니다. (적절함)
BASE_SHAKE_INTENSITY = 5.0 
# ==========================================

def set_rendering(enable):
    """PyBullet 렌더링 On/Off 제어 (속도 최적화 핵심)"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if enable else 0)
    # GUI 업데이트 자체를 멈추는 옵션 (선택적)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if enable else 0)

def wait_for_settle(item_ids, velocity_threshold=0.05, max_steps=2000, vis_mode=0):
    """
    안정화 대기 함수
    vis_mode가 1 또는 2인 경우 렌더링을 꺼서 속도를 높임
    """
    # [Vis 1, 2] 안정화 계산 중에는 렌더링 끄기 (속도 향상)
    if vis_mode >= 1:
        set_rendering(False)

    settled = False
    for i in range(max_steps):
        p.stepSimulation()
        
        if i % 20 == 0: # 20스텝마다 검사
            max_vel = 0.0
            is_moving = False
            for uid in item_ids:
                lin_vel, _ = p.getBaseVelocity(uid)
                v = np.linalg.norm(lin_vel)
                if v > max_vel: max_vel = v
                if v > velocity_threshold:
                    is_moving = True
                    break 
            
            if not is_moving:
                settled = True
                break
    
    # [Vis 1, 2] 안정화 끝나면 다시 켤지는 호출자가 결정하지만,
    # Vis 2(Adaptive)의 경우 상태 확인을 위해 잠깐 켤 수 있음.
    # 여기서는 상태 유지만 하고 함수 종료.
    if vis_mode >= 1:
        # Vis 2라면 끝나고 잠깐 화면 갱신을 위해 켜줄 수 있음
        if vis_mode == 2:
            set_rendering(True)
            time.sleep(0.01) # 화면 그릴 시간 확보
            set_rendering(False) 

    if not settled:
        # print(f"   [안정화] 시간 초과 (Max Vel: {max_vel:.3f})")
        pass

def get_pile_status(item_ids, half_w_m):
    if not item_ids: return 0.0, 0
    max_z = 0.0
    valid_count = 0
    boundary = half_w_m * 1.1
    
    for b_id in item_ids:
        pos, _ = p.getBasePositionAndOrientation(b_id)
        if abs(pos[0]) > boundary or abs(pos[1]) > boundary:
            continue
        valid_count += 1
        if pos[2] > max_z:
            max_z = pos[2]
            
    return max_z, valid_count

def run_fast_physics_packing(filename, container_size, goal_height, vis_mode=0):
    """
    vis_mode:
      0: Full Visualization (Real-time)
      1: Final Result Only (Fastest)
      2: Adaptive (Batch update only)
    """
    
    # 단위 변환 (mm -> m)
    W_mm, L_mm, H_mm = container_size
    W_m, L_m, H_m = W_mm*0.001, L_mm*0.001, H_mm*0.001
    GOAL_HEIGHT_m = goal_height * 0.001
    STL_FILENAME = filename

    # 1. PyBullet 연결
    try: p.disconnect()
    except: pass
    
    p.connect(p.GUI) # GUI 모드로 열되, 렌더링은 set_rendering으로 제어
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=20)
    
    # 카메라 설정
    p.resetDebugVisualizerCamera(
        cameraDistance=max(W_m, H_m) * 1.5, 
        cameraYaw=45, cameraPitch=-30, 
        cameraTargetPosition=[0, 0, H_m/3]
    )

    # 초기 렌더링 설정
    if vis_mode == 1:
        print("   [Vis] Mode 1: 렌더링을 끕니다. (최종 결과만 표시)")
        set_rendering(False)
    elif vis_mode == 2:
        print("   [Vis] Mode 2: 적응형 렌더링 (중간 과정 생략)")

    # 2. 환경 생성
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
        if np.max(mesh.extents) > 1.0:
            scale_factor = [0.001, 0.001, 0.001]
        else:
            scale_factor = [1.0, 1.0, 1.0]
            
        col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=STL_FILENAME, meshScale=scale_factor)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=STL_FILENAME, meshScale=scale_factor)
    except Exception as e:
        print(f"[Error] STL 로드 실패: {e}")
        return

    item_ids = []
    current_height_m = 0.0
    
    print(f"=== 물리 시뮬레이션 시작 (Goal: {goal_height}mm) ===")
    start_time = time.time()

    while len(item_ids) < MAX_TOTAL_ITEMS:
        current_height_m, valid_count = get_pile_status(item_ids, half_w)
        
        if current_height_m >= GOAL_HEIGHT_m:
            print(f"\n[성공] 목표 높이 도달: {current_height_m * 1000:.1f}mm")
            break
            
        drop_z_start = max(current_height_m, 0.05) + W_m * 0.5
        
        # [Vis 2] 투하 및 계산 중에는 렌더링 끄기
        if vis_mode == 2: set_rendering(False)
        
        # A. 투하
        for _ in range(BATCH_SIZE):
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

        # C. [Adaptive Shake] 적응형 흔들기
        # 높이가 높아질수록(progress -> 1.0), damping이 커져서 힘이 약해짐 (최소 20% 파워 유지)
        progress = min(current_height_m / GOAL_HEIGHT_m, 1.0)
        damping_factor = max(0.2, 1.0 - progress) 
        
        current_shake_force = BASE_SHAKE_INTENSITY * damping_factor
        
        # Vis 0(Full)일 때만 흔드는 과정 보여줌. 1, 2는 생략(계산만 수행)
        for t in range(30):
            fx = (random.random() - 0.5) * current_shake_force
            fy = (random.random() - 0.5) * current_shake_force
            for uid in item_ids:
                 p.applyExternalForce(uid, -1, [fx, fy, -2.0], [0,0,0], p.WORLD_FRAME)
            p.stepSimulation()
            if vis_mode == 0: time.sleep(0.001)

        # D. 최종 안정화
        wait_for_settle(item_ids, velocity_threshold=0.005, max_steps=800, vis_mode=vis_mode)

        # 상태 출력 및 렌더링 업데이트
        current_height_m, valid_count = get_pile_status(item_ids, half_w)
        
        # [Vis 2] 배치 완료 시점에만 화면 갱신
        if vis_mode == 2:
            set_rendering(True)
            p.stepSimulation() # 한 프레임 그려줌
            # time.sleep(0.01) # 필요시 딜레이

        print(f"   [진행] {len(item_ids)}개 투입 | 높이: {current_height_m*1000:.1f}mm | Shake: {current_shake_force:.2f}N")

    # [Vis 1] 종료 시점에 렌더링 켜기
    if vis_mode == 1:
        set_rendering(True)
        
    print(f"\n>>> 시뮬레이션 종료. (총 시간: {time.time()-start_time:.1f}초)")
    print(">>> 창을 닫으면 프로그램이 종료됩니다.")
    
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    run_fast_physics_packing("llamp_mod.stl", [300, 300, 400], 115.0, vis_mode=0)