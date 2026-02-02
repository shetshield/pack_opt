import numpy as np
import trimesh
import time
import os

class SmartVoxelPacker:
    def __init__(self, grid_shape, voxel_pitch, goal_z_voxel):
        self.shape = grid_shape
        self.pitch = voxel_pitch
        self.goal_z = goal_z_voxel
        self.grid = np.zeros(grid_shape, dtype=bool)
        self.items_count = 0

    def load_item(self, filename):
        mesh = trimesh.load(filename)
        # 단위 통일 (m -> mm)
        if mesh.extents.max() < 1.0: mesh.apply_scale(1000)
        
        vox = mesh.voxelized(pitch=self.pitch).matrix
        return mesh, vox

    def check_collision(self, item_vox, x, y, z):
        idx, idy, idz = item_vox.shape
        if x + idx > self.shape[0] or y + idy > self.shape[1] or z + idz > self.shape[2]:
            return True
        view = self.grid[x:x+idx, y:y+idy, z:z+idz]
        return np.any(view & item_vox)

    def pack(self, item_vox):
        idx, idy, idz = item_vox.shape
        range_x = self.shape[0] - idx + 1
        range_y = self.shape[1] - idy + 1
        
        for z in range(self.goal_z - idz + 1):
            for x in range(range_x):
                for y in range(range_y):
                    if not self.check_collision(item_vox, x, y, z):
                        self.grid[x:x+idx, y:y+idy, z:z+idz] |= item_vox
                        self.items_count += 1
                        return True
        return False

def run_voxel_packing(filename, container_size, goal_height):
    W, L, H = container_size
    PITCH = 2.0 # mm
    
    grid_shape = (int(W/PITCH), int(L/PITCH), int(H/PITCH))
    goal_voxel = int(goal_height/PITCH)
    
    print(f"  - Grid Shape: {grid_shape}, Pitch: {PITCH}mm")
    
    packer = SmartVoxelPacker(grid_shape, PITCH, goal_voxel)
    mesh, vox = packer.load_item(filename)
    
    start = time.time()
    while True:
        success = packer.pack(vox)
        if not success: break
        if packer.items_count % 10 == 0:
            print(f"    Packed: {packer.items_count}...")
            
    print(f"\n>>> Final Result: {packer.items_count} items packed.")
    print(f"    Time: {time.time() - start:.2f} sec")