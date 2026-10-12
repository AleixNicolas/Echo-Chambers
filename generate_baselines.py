import json
import random
import math
from pathlib import Path
import config
from src import networks

def generate_item_pool():
    leans = {'L': 0.1, 'LL': 0.3, 'C': 0.5, 'LR': 0.7, 'R': 0.9}
    pool = {}
    for source in range(1, 9):
        for l_str, l_val in leans.items():
            for item_num in range(1, 5):
                pool[f"S{source}-{l_str}-{item_num}"] = l_val
    return pool

def generate_all_baselines():
    print("=== GENERATING NETWORK BASELINES ===")
    config.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    
    global_pool = generate_item_pool()
    extreme_frac = getattr(config, 'BASE_TOPIC_EXTREME_FRACTION', 0.5)
    
    for n in config.N_VALUES:
        for k in config.K_VALUES:
            for cross_edge in config.PPM_CROSS_EDGE_FRACTIONS:
                print(f"-> Generating Baseline N={n}, K={k}, Cross={cross_edge}...")
                
                # 1. GENERATE THE SEGREGATED TOPOLOGY
                G_seg = networks.generate_ppm_topology(n, k, cross_edge)
                adj_seg = {str(node): [str(nb) for nb in G_seg.neighbors(node)] for node in G_seg.nodes()}
                
                # 2. REWIRE TO INTEGRATED TOPOLOGY (Configuration Model via Edge Swaps)
                G_int = networks.rewire_to_integrated(G_seg)
                adj_int = {str(node): [str(nb) for nb in G_int.neighbors(node)] for node in G_int.nodes()}
                
                half = n // 2
                
                # 3. GENERATE EXACT OPINION POOLS
                num_extreme = int(math.floor(half * extreme_frac))
                num_moderate = half - num_extreme
                
                left_pool = sorted([0.1] * num_extreme + [0.3] * num_moderate)
                right_pool = sorted([0.9] * num_extreme + [0.7] * num_moderate)
                
                random.shuffle(left_pool)
                random.shuffle(right_pool)
                
                # 4. ASSIGN OPINIONS TO NODES (Chamber 1 = Left, Chamber 2 = Right)
                # This assignment will be used for BOTH regimes to ensure perfectly identical nodes
                node_ops = {}
                for i in range(half):
                    node_ops[i] = left_pool[i]
                for i in range(half, n):
                    node_ops[i] = right_pool[i - half]

                # Build the node dictionary once so opinions and starting_items are identical
                def build_nodes_dict(ops_dict):
                    nodes_dict = {}
                    for node in range(n):
                        start_items = random.sample(list(global_pool.keys()), 4)
                        nodes_dict[str(node)] = {
                            "opinion": ops_dict[node],
                            "starting_items": start_items
                        }
                    return nodes_dict

                shared_nodes_dict = build_nodes_dict(node_ops)

                # 5. ASSEMBLE BASELINE DATA
                baseline_data = {
                    "global_item_pool": global_pool,
                    "segregated_baseline": {
                        "nodes": shared_nodes_dict,
                        "network": adj_seg
                    },
                    "integrated_baseline": {
                        "nodes": shared_nodes_dict,
                        "network": adj_int
                    }
                }
                
                out_path = config.get_baseline_path(n, k)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(baseline_data, f, indent=4)
                    
                networks.plot_simulated_network(out_path, config.BASELINES_DIR)
                
    print("\n=== GENERATION COMPLETE ===")

if __name__ == "__main__":
    generate_all_baselines()