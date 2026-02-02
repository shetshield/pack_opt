from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MixedLayerOptimizer:
    def __init__(self, container_size):
        # container_size (mm)
        self.W, self.L, self.physical_H = container_size
    
    def solve_2d_layer(self, item_w, item_l, time_limit=5.0):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        
        # mm 단위 계산
        max_items = int((self.W * self.L) / (item_w * item_l))
        candidate_count = min(max_items, int(max_items * 1.0))
        
        x, y, l_var, w_var, is_packed = [], [], [], [], []
        rotations = [(item_w, item_l), (item_l, item_w), (0, 0)]
        
        for i in range(candidate_count):
            packed = model.NewBoolVar(f'packed_{i}')
            is_packed.append(packed)
            xi = model.NewIntVar(0, int(self.W), f'x_{i}')
            yi = model.NewIntVar(0, int(self.L), f'y_{i}')
            li = model.NewIntVar(0, int(max(item_w, item_l)), f'l_{i}')
            wi = model.NewIntVar(0, int(max(item_w, item_l)), f'w_{i}')
            
            x.append(xi); y.append(yi); l_var.append(li); w_var.append(wi)
            
            model.AddAllowedAssignments([li, wi], rotations)
            model.Add(li == 0).OnlyEnforceIf(packed.Not())
            model.Add(wi == 0).OnlyEnforceIf(packed.Not())
            model.Add(li > 0).OnlyEnforceIf(packed)
            model.Add(xi + li <= int(self.W)).OnlyEnforceIf(packed)
            model.Add(yi + wi <= int(self.L)).OnlyEnforceIf(packed)
            
            if i > 0: model.Add(packed <= is_packed[i-1])

        # No Overlap
        for i in range(candidate_count):
            for j in range(i + 1, candidate_count):
                left = model.NewBoolVar(f'{i}L{j}')
                right = model.NewBoolVar(f'{i}R{j}')
                above = model.NewBoolVar(f'{i}A{j}')
                below = model.NewBoolVar(f'{i}B{j}')
                
                model.Add(x[i] + l_var[i] <= x[j]).OnlyEnforceIf(left)
                model.Add(x[j] + l_var[j] <= x[i]).OnlyEnforceIf(right)
                model.Add(y[i] + w_var[i] <= y[j]).OnlyEnforceIf(below)
                model.Add(y[j] + w_var[j] <= y[i]).OnlyEnforceIf(above)
                model.AddBoolOr([left, right, above, below]).OnlyEnforceIf([is_packed[i], is_packed[j]])
        
        model.Maximize(sum(is_packed))
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)
        
        items_data = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            count = int(solver.Value(sum(is_packed)))
            for i in range(candidate_count):
                if solver.Value(is_packed[i]):
                    items_data.append({
                        'x': solver.Value(x[i]), 'y': solver.Value(y[i]),
                        'lx': solver.Value(l_var[i]), 'ly': solver.Value(w_var[i])
                    })
            return count, items_data
        return 0, []

    def optimize_mixed_stacking(self, item_dims, goal_height):
        print(f"  - Goal Height: {goal_height}mm")
        
        d1, d2, d3 = sorted(item_dims)
        layer_types = [
            {'h': d1, 'w': d2, 'l': d3, 'name': f"Flat({d1:.1f})"},
            {'h': d2, 'w': d1, 'l': d3, 'name': f"Side({d2:.1f})"},
            {'h': d3, 'w': d1, 'l': d2, 'name': f"Upright({d3:.1f})"}
        ]
        
        valid_layers = []
        for l_type in layer_types:
            if l_type['h'] > goal_height: continue
            print(f"  - Analyzing Layer [{l_type['name']}]...", end=" ", flush=True)
            count, data = self.solve_2d_layer(l_type['w'], l_type['l'], time_limit=5.0)
            print(f"-> Capacity: {count} items")
            if count > 0:
                l_type['count'] = count
                l_type['data'] = data
                valid_layers.append(l_type)

        if not valid_layers:
            print("  [Fail] No items fit.")
            return

        print("  - Solving Knapsack for Layer Combination...")
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        
        min_h = min(l['h'] for l in valid_layers)
        max_possible_layers = int(goal_height // min_h) + 1
        
        layer_counts = []
        for i, l_type in enumerate(valid_layers):
            c = model.NewIntVar(0, max_possible_layers, f'count_{i}')
            layer_counts.append(c)

        total_height = sum(layer_counts[i] * int(valid_layers[i]['h']) for i in range(len(valid_layers)))
        model.Add(total_height <= int(goal_height))

        total_items = sum(layer_counts[i] * valid_layers[i]['count'] for i in range(len(valid_layers)))
        model.Maximize(total_items)

        solver.Solve(model)
        
        final_total = int(solver.Value(total_items))
        used_h = int(solver.Value(total_height))
        
        print(f"\n>>> Final Result: {final_total} items (Height Used: {used_h}/{goal_height}mm)")
        
        stack_plan = []
        for i, l_type in enumerate(valid_layers):
            qty = int(solver.Value(layer_counts[i]))
            if qty > 0:
                print(f"    - {l_type['name']} x {qty} layers")
                for _ in range(qty): stack_plan.append(l_type)
        
        self.visualize(stack_plan, goal_height, final_total)

    def visualize(self, stack_plan, goal_height, total_items):
        if not stack_plan: return
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        W, L, H = self.W, self.L, self.physical_H
        ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], [0, 0, 0, 0, 0], 'k-', lw=1)
        ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], [H, H, H, H, H], 'k-', lw=0.1)
        for x, y in [(0,0), (W,0), (W,L), (0,L)]:
            ax.plot([x, x], [y, y], [0, H], 'k-', lw=0.1)
            
        xx, yy = np.meshgrid([0, W], [0, L])
        zz = np.full_like(xx, goal_height)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='r')

        current_z = 0
        colors = ['blue', 'green', 'orange']
        
        for layer in stack_plan:
            layer_h = layer['h']
            c = colors[int(layer_h) % 3] 
            
            for item in layer['data']:
                ax.bar3d(item['x'], item['y'], current_z, 
                         item['lx'], item['ly'], layer_h, 
                         color=c, alpha=0.8, edgecolor='k', linewidth=0.5)
            current_z += layer_h
            
        ax.set_xlim(0, max(W,L,H)); ax.set_ylim(0, max(W,L,H)); ax.set_zlim(0, max(W,L,H))
        plt.title(f"Optimized Stack: {total_items} Items")
        plt.show()