import trimesh
import numpy as np

class ShapeAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.mesh = trimesh.load(filepath)
        
        # [단위 통일] m -> mm 변환
        if self.mesh.extents.max() < 1.0:
            print("  [Info] STL unit conversion (m -> mm) applied.")
            self.mesh.apply_scale(1000)
            
    def analyze(self):
        vol_mesh = self.mesh.volume
        
        try: obb = self.mesh.bounding_box_oriented
        except: obb = self.mesh.bounding_box
        
        vol_obb = obb.volume
        hull = self.mesh.convex_hull
        vol_hull = hull.volume
        
        rectangularity = vol_mesh / vol_obb if vol_obb > 0 else 0
        solidity = vol_mesh / vol_hull if vol_hull > 0 else 0
        dims = self.mesh.extents # (mm 단위)
        
        print(f"  - Dimensions (mm): {np.round(dims, 1)}")
        print(f"  - Rectangularity: {rectangularity:.3f}")
        print(f"  - Solidity: {solidity:.3f}")
        
        # 엔진 추천 로직
        if rectangularity > 0.8:
            return "MATH_OPTIMIZER", dims
        elif solidity > 0.8:
            return "VOXEL_PACKER", dims
        else:
            return "PHYSICS_SIM", dims

if __name__ == "__main__":
    analyzer = ShapeAnalyzer("llamp_mod.stl")
    print(analyzer.analyze())
