import os
import json
import random
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
    os.makedirs(config.BASELINES_DIR, exist_ok=True)
    
    global_pool = generate_item_pool()
    
    for n in config.N_VALUES:
        for k in config.K_VALUES:
            for cross_edge in config.PPM_CROSS_EDGE_FRACTIONS:
                print(f"-> Generating Baseline N={n}, K={k}, Cross={cross_edge}...")
                
                # Generate Graph
                G = networks.generate_ppm_topology(n, k, cross_edge)
                adj_list = {str(node): [str(nb) for nb in G.neighbors(node)] for node in G.nodes()}
                
                # Generate Segregated Opinions
                seg_ops = {}
                half = n // 2
                for node in range(n):
                    if node < half:
                        seg_ops[node] = random.choice([0.1, 0.3])
                    else:
                        seg_ops[node] = random.choice([0.7, 0.9])
                        
                # Generate Integrated Opinions (Shuffle the segregated ones)
                int_ops_list = list(seg_ops.values())
                random.shuffle(int_ops_list)
                int_ops = {node: int_ops_list[node] for node in range(n)}
                
                def build_nodes_dict(ops_dict):
                    nodes_dict = {}
                    for node in range(n):
                        # Sample 4 random starting items per node
                        start_items = random.sample(list(global_pool.keys()), 4)
                        nodes_dict[str(node)] = {
                            "opinion": ops_dict[node],
                            "starting_items": start_items
                        }
                    return nodes_dict

                baseline_data = {
                    "global_item_pool": global_pool,
                    "segregated_baseline": {
                        "nodes": build_nodes_dict(seg_ops),
                        "network": adj_list
                    },
                    "integrated_baseline": {
                        "nodes": build_nodes_dict(int_ops),
                        "network": adj_list
                    }
                }
                
                out_path = config.get_baseline_path(n, k)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(baseline_data, f, indent=4)
                    
                # Plot the generated networks
                networks.plot_simulated_network(out_path, config.BASELINES_DIR)
                
    print("\n=== GENERATION COMPLETE ===")

if __name__ == "__main__":
    generate_all_baselines()