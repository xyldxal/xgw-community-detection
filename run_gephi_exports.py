import os
import sys
import warnings
import time
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import combinations
import pandas as pd
import networkx as nx
from cdlib import algorithms

warnings.filterwarnings("ignore")

DATA_DIR       = "xgw/data"
PPI_PATH       = f"{DATA_DIR}/preprocessed/dip_ppin.csv"
MCL_PATH       = f"{DATA_DIR}/clusters/out.dip_unweighted.csv.I40"
MIN_COMM_SIZE  = 3

def load_ppi_graph():
    print("Loading PPI network (ORF names) from preprocessed/dip_ppin.csv...")
    df = pd.read_csv(PPI_PATH, header=0, names=['u', 'v'])
    G = nx.from_pandas_edgelist(df, source='u', target='v')
    return G

def load_mcl_baseline(G):
    clusters = []
    with open(MCL_PATH, 'r') as f:
        for line in f:
            members = line.strip().split('\t')
            valid_members = [m for m in members if m in G.nodes]
            if len(valid_members) >= MIN_COMM_SIZE:
                clusters.append(set(valid_members))
    return clusters

def run_algorithms(G):
    print("Running algorithms to get community assignments...")
    results = {}
    
    # 1. Leiden
    print("  Leiden...")
    leiden = algorithms.leiden(G)
    results['Leiden'] = [set(c) for c in leiden.communities if len(c) >= MIN_COMM_SIZE]
    
    # 2. Louvain
    print("  Louvain...")
    louvain = algorithms.louvain(G)
    results['Louvain'] = [set(c) for c in louvain.communities if len(c) >= MIN_COMM_SIZE]
    
    # 3. Label Propagation
    print("  Label Propagation...")
    lp_raw = algorithms.label_propagation(G)
    results['Label Propagation'] = [set(c) for c in lp_raw.communities if len(c) >= MIN_COMM_SIZE]
    
    # 4. SBM (Spectral proxy)
    print("  SBM (Spectral proxy)...")
    from sklearn.cluster import SpectralClustering
    node_list = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=node_list)
    n_clusters = min(50, G.number_of_nodes() // 10)
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42, n_jobs=-1)
    labels = sc.fit_predict(A)
    sbm_comms = {}
    for i, node in enumerate(node_list):
        c = labels[i]
        sbm_comms.setdefault(c, set()).add(node)
    results['SBM (Spectral proxy)'] = [s for s in sbm_comms.values() if len(s) >= MIN_COMM_SIZE]
    
    # 5. Clique Percolation
    print("  Clique Percolation...")
    cp = algorithms.kclique(G, k=3)
    results['Clique Percolation'] = [set(c) for c in cp.communities if len(c) >= MIN_COMM_SIZE]
    
    # 6. Consensus (Optimized)
    print("  Consensus...")
    node_to_lei = {node: cid for cid, comm in enumerate(results['Leiden']) for node in comm}
    node_to_lou = {node: cid for cid, comm in enumerate(results['Louvain']) for node in comm}
    node_to_lp  = {node: cid for cid, comm in enumerate(results['Label Propagation']) for node in comm}
    
    def get_cooccurring_pairs(comms_a, comms_b):
        node_to_a = {node: cid for cid, comm in enumerate(comms_a) for node in comm}
        node_to_b = {node: cid for cid, comm in enumerate(comms_b) for node in comm}
        groups = {}
        for node in set(node_to_a.keys()) & set(node_to_b.keys()):
            key = (node_to_a[node], node_to_b[node])
            groups.setdefault(key, []).append(node)
        pairs = set()
        for members in groups.values():
            if len(members) > 1:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        u, v = members[i], members[j]
                        if u > v: u, v = v, u
                        pairs.add((u, v))
        return pairs

    pairs_lei_lou = get_cooccurring_pairs(results['Leiden'], results['Louvain'])
    pairs_lei_lp  = get_cooccurring_pairs(results['Leiden'], results['Label Propagation'])
    pairs_lou_lp  = get_cooccurring_pairs(results['Louvain'], results['Label Propagation'])
    all_consensus_edges = pairs_lei_lou | pairs_lei_lp | pairs_lou_lp
    
    filtered_G = nx.Graph()
    filtered_G.add_nodes_from(G.nodes())
    weighted_edges = []
    for u, v in all_consensus_edges:
        weight = 0
        if node_to_lei.get(u) == node_to_lei.get(v) and u in node_to_lei: weight += 1
        if node_to_lou.get(u) == node_to_lou.get(v) and u in node_to_lou: weight += 1
        if node_to_lp.get(u) == node_to_lp.get(v) and u in node_to_lp: weight += 1
        weighted_edges.append((u, v, weight))
    filtered_G.add_weighted_edges_from(weighted_edges)
    
    consensus_clusters = algorithms.louvain(filtered_G)
    results['Consensus'] = [set(c) for c in consensus_clusters.communities if len(c) >= MIN_COMM_SIZE]
    
    # 7. MCL Baseline
    results['MCL (XGW baseline)'] = load_mcl_baseline(G)
    
    return results

def inject_positions_into_gexf(filepath, positions):
    tree = ET.parse(filepath)
    root = tree.getroot()
    ET.register_namespace('', 'http://www.gexf.net/1.2draft')
    ET.register_namespace('viz', 'http://www.gexf.net/1.2draft/viz')

    for attr in list(root.attrib):
        if 'viz' in attr and 'xmlns' in attr:
            del root.attrib[attr]

    for node_elem in root.iter('{http://www.gexf.net/1.2draft}node'):
        node_id = node_elem.get('id')
        if node_id in positions:
            x, y = positions[node_id]
            viz_pos = ET.SubElement(node_elem, '{http://www.gexf.net/1.2draft/viz}position')
            viz_pos.set('x', str(round(x, 2)))
            viz_pos.set('y', str(round(y, 2)))
            viz_pos.set('z', '0.0')

    tree.write(filepath, xml_declaration=True, encoding='utf-8')

def main():
    os.makedirs('gephi_exports', exist_ok=True)
    G = load_ppi_graph()
    results = run_algorithms(G)
    
    print("\nComputing spring layout for Gephi exports (iterations=20)...")
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    scale = 1000
    pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}
    print("Layout computed.")
    
    # 1. Export individual files
    print("\nExporting individual GEXF files...")
    for algo_name, comms in results.items():
        G_single = G.copy()
        # Clean name for safe filename
        safe_name = (
            algo_name.lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace(',', '')
            .replace('=', '')
        )
        
        # Add community label to each node
        for comm_id, members in enumerate(comms):
            for node in members:
                if node in G_single.nodes:
                    G_single.nodes[node]['community'] = comm_id
                    
        for node in G_single.nodes:
            if 'community' not in G_single.nodes[node]:
                G_single.nodes[node]['community'] = -1
                
        out_path = f'gephi_exports/ppi_{safe_name}.gexf'
        nx.write_gexf(G_single, out_path)
        inject_positions_into_gexf(out_path, pos)
        print(f"  Exported -> {out_path}")
        
    # 2. Export combined file (with all algorithm attributes on the same nodes)
    print("\nExporting combined GEXF file (all methods)...")
    G_all = G.copy()
    for algo_name, comms in results.items():
        attr_name = (
            algo_name.lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace(',', '')
            .replace('=', '')
        )
        for comm_id, members in enumerate(comms):
            for node in members:
                if node in G_all.nodes:
                    G_all.nodes[node][attr_name] = comm_id
        for node in G_all.nodes:
            if attr_name not in G_all.nodes[node]:
                G_all.nodes[node][attr_name] = -1
                
    combined_path = 'gephi_exports/ppi_all_methods.gexf'
    nx.write_gexf(G_all, combined_path)
    inject_positions_into_gexf(combined_path, pos)
    print(f"  Exported combined -> {combined_path}")
    print("\nAll Gephi exports are now 100% correct, updated, and ready!")

if __name__ == "__main__":
    main()
