import networkx as nx
import random
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config

def generate_ppm_topology(n_nodes, k, cross_edge_fraction):
    """Generates the Planted Partition Model graph."""
    if n_nodes % 2 != 0: raise ValueError("N must be even.")
    half_n = n_nodes // 2
    nodes_b = list(range(half_n, n_nodes))
    
    graph_a = nx.random_regular_graph(k, half_n)
    graph_b = nx.random_regular_graph(k, half_n)
    graph_b = nx.relabel_nodes(graph_b, {i: nodes_b[i] for i in range(half_n)})
    combined_graph = nx.compose(graph_a, graph_b)
    
    num_swaps = int(round((n_nodes * k / 2) * cross_edge_fraction)) // 2
    edges_a, edges_b = list(graph_a.edges()), list(graph_b.edges())
    random.shuffle(edges_a); random.shuffle(edges_b)
    
    swaps_completed, bridged_nodes = 0, set()
    for u, v in edges_a:
        if swaps_completed >= num_swaps: break
        if u in bridged_nodes or v in bridged_nodes: continue
        for x, y in edges_b:
            if x in bridged_nodes or y in bridged_nodes: continue
            if not combined_graph.has_edge(u, x) and not combined_graph.has_edge(v, y):
                combined_graph.remove_edge(u, v); combined_graph.remove_edge(x, y)
                combined_graph.add_edge(u, x); combined_graph.add_edge(v, y)
                bridged_nodes.update([u, v, x, y])
                swaps_completed += 1
                edges_b.remove((x, y))
                break
    return combined_graph

def plot_simulated_network(baseline_path, out_dir):
    """Visualizes the Segregated vs Integrated architectures."""
    if not os.path.exists(baseline_path): return
    with open(baseline_path, 'r') as f: data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    val_to_pos = {v: k for k, v in config.POS_MIDPOINTS.items()}
    
    for idx, struct in enumerate(["segregated_baseline", "integrated_baseline"]):
        ax = axes[idx]
        G = nx.Graph()
        
        for source, targets in data[struct]['network'].items():
            for t in targets: G.add_edge(int(source), int(t))
            
        node_colors = []
        for n in sorted(G.nodes()):
            op = data[struct]['nodes'][str(n)]['opinion']
            closest_pos = min(val_to_pos.keys(), key=lambda k: abs(k - op))
            node_colors.append(config.POS_COLORS[val_to_pos[closest_pos]])
            
        half_n = len(G.nodes()) // 2
        intra_edges = [(u, v) for u, v in G.edges() if (u < half_n and v < half_n) or (u >= half_n and v >= half_n)]
        cross_edges = [(u, v) for u, v in G.edges() if (u < half_n and v >= half_n) or (u >= half_n and v < half_n)]
        
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=ax, alpha=0.6, edge_color='grey')
        nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax, alpha=0.3, edge_color='grey', width=2.5)
        nx.draw_networkx_nodes(G, pos, nodelist=sorted(G.nodes()), ax=ax, node_color=node_colors, node_size=500, edgecolors='white', linewidths=1.5)
        
        ax.set_title("Segregated Topology" if "segregated" in struct else "Integrated Topology", fontsize=16, fontweight='bold')
        ax.axis('off')
        
    legend_handles = [mpatches.Patch(color=config.POS_COLORS[p], label=config.POS_LABELS[p]) for p in [5, 4, 3, 2, 1]]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=14, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    filename = os.path.basename(baseline_path).replace('.json', '.pdf')
    plt.savefig(os.path.join(out_dir, f"Topology_{filename}"), dpi=300)
    plt.close(fig)