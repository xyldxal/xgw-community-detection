import os
import sys
import warnings
import time
from collections import Counter
from itertools import combinations
import pandas as pd
import networkx as nx
from cdlib import algorithms

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR       = "xgw/data"
PPI_PATH       = f"{DATA_DIR}/preprocessed/dip_ppin.csv"
CYC_PATH       = f"{DATA_DIR}/swc/complexes_CYC.txt"
MIN_COMM_SIZE  = 3

SPOTLIGHT_NAMES = [
    "20S proteasome",
    "DNA-directed RNA polymerase II complex",
    "Kornberg's mediator (SRB) complex",
]

def load_ppi_graph():
    print("Loading DIP PPI network...")
    df = pd.read_csv(PPI_PATH, header=0, names=['u', 'v'])
    G = nx.from_pandas_edgelist(df, source='u', target='v')
    print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    return G

def load_cyc2008():
    df = pd.read_csv(
        CYC_PATH, sep='\t', header=None,
        names=['protein', 'complex_id', 'complex_name']
    )
    complexes = {}
    for _, group in df.groupby('complex_id'):
        name   = group['complex_name'].iloc[0].strip().strip('"')
        prots  = set(group['protein'].tolist())
        complexes[name] = prots
    return complexes

def load_mcl_at_inflation(inflation_str, G):
    filepath = f"{DATA_DIR}/clusters/out.dip_unweighted.csv.{inflation_str}"
    clusters = []
    if not os.path.exists(filepath):
        print(f"  WARNING: baseline file '{filepath}' not found.")
        return clusters
    with open(filepath, 'r') as f:
        for line in f:
            members = line.strip().split('\t')
            valid_members = [m for m in members if m in G.nodes]
            if len(valid_members) >= MIN_COMM_SIZE:
                clusters.append(set(valid_members))
    return clusters

def run_algorithms(G):
    print("Running algorithms...")
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
    import numpy as np
    from sklearn.cluster import SpectralClustering
    node_list = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=node_list)
    n_clusters = min(50, G.number_of_nodes() // 10)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )
    labels = sc.fit_predict(A)
    sbm_comms = {}
    for i, node in enumerate(node_list):
        c = labels[i]
        sbm_comms.setdefault(c, set()).add(node)
    results['SBM (Spectral proxy)'] = [
        s for s in sbm_comms.values() if len(s) >= MIN_COMM_SIZE
    ]
    
    # 5. Clique Percolation (k=3)
    print("  Clique Percolation...")
    cp = algorithms.kclique(G, k=3)
    results['Clique Percolation'] = [set(c) for c in cp.communities if len(c) >= MIN_COMM_SIZE]
    
    # 6. Consensus (Optimized)
    print("  Consensus (Optimized)...")
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
    
    # 7. MCL (loaded from inflation I40, which matches our paper baseline)
    results['MCL (XGW baseline)'] = load_mcl_at_inflation("I40", G)
    
    return results

def calculate_pairwise_retention(results, G):
    print("\nCalculating pairwise retention rates (Phase 2)...")
    inflations = ["I20", "I30", "I40", "I50"]
    labels = ["I=2.0", "I=3.0", "I=4.0", "I=5.0"]
    
    algorithms_to_eval = [
        'Leiden', 'Louvain', 'Label Propagation', 
        'SBM (Spectral proxy)', 'Clique Percolation', 'Consensus', 'MCL (XGW baseline)'
    ]
    
    retention_table = []
    
    for infl_str, lbl in zip(inflations, labels):
        print(f"  Processing inflation baseline {lbl}...")
        baseline_clusters = load_mcl_at_inflation(infl_str, G)
        
        # Extract unique pairs
        baseline_pairs = set()
        for cluster in baseline_clusters:
            for u, v in combinations(cluster, 2):
                if u > v: u, v = v, u
                baseline_pairs.add((u, v))
                
        total_pairs = len(baseline_pairs)
        print(f"    Total unique pairs in baseline {lbl}: {total_pairs:,}")
        
        column_rates = {}
        for algo_name in algorithms_to_eval:
            comms = results.get(algo_name, [])
            # Map node to community ID
            node_to_comm = {}
            for cid, comm in enumerate(comms):
                for node in comm:
                    node_to_comm[node] = cid
                    
            retained = 0
            for u, v in baseline_pairs:
                if u in node_to_comm and v in node_to_comm:
                    if node_to_comm[u] == node_to_comm[v]:
                        retained += 1
            rate = (retained / total_pairs) * 100.0 if total_pairs > 0 else 0.0
            column_rates[algo_name] = rate
            print(f"      {algo_name:<25}: {rate:.2f}% retained ({retained:,} pairs)")
            
        retention_table.append({
            'Inflation': lbl,
            'Total Pairs': total_pairs,
            **column_rates
        })
        
    df = pd.DataFrame(retention_table)
    return df

def find_best_swallowing_community(complex_prots, communities):
    """Find the community that contains the most proteins of complex_prots."""
    best_c_idx = -1
    best_overlap = 0
    best_c_size = 0
    best_c_set = set()
    
    for idx, c in enumerate(communities):
        overlap = len(complex_prots & c)
        if overlap > best_overlap:
            best_overlap = overlap
            best_c_idx = idx
            best_c_size = len(c)
            best_c_set = c
            
    return best_c_idx, best_overlap, best_c_size, best_c_set

def generate_phase3_reports(results, cyc_complexes):
    print("\nExtracting coarse community mapping details (Phase 3)...")
    lines = []
    w = lines.append
    
    w("==============================================================================")
    w("PHASE 3: COARSE COMMUNITY LUMPING / OVER-PARTITIONING MAPPINGS")
    w("         For Marxel to detail in Section VII-C (Biological Discussion)")
    w("==============================================================================")
    w("")
    w("We analyze three major yeast complexes from CYC2008 ground-truth and trace")
    w("how they are clustered (or lumped into giant hairball communities) by the")
    w("different graph-based algorithms compared to MCL (XGW baseline).")
    w("")
    
    for name in SPOTLIGHT_NAMES:
        if name not in cyc_complexes:
            continue
        prots = cyc_complexes[name]
        w("-" * 78)
        w(f"Complex: {name}")
        w(f"Ground-Truth Size in CYC2008: {len(prots)} proteins")
        w("-" * 78)
        
        # Print proteins in this complex (sorted)
        w(f"Proteins: {', '.join(sorted(list(prots)))}")
        w("")
        w(f"  {'Algorithm':<28} {'Subunits Found':<16} {'Best Comm Size':<16} {'Behavior / Community ID'}")
        w(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*24}")
        
        for algo_name, comms in results.items():
            if algo_name.startswith('__'): continue
            
            c_idx, overlap, c_size, c_set = find_best_swallowing_community(prots, comms)
            
            if overlap == 0:
                behavior = "COMPLETELY SPLIT (0 overlap)"
            elif overlap == len(prots):
                if c_size > len(prots) * 3:
                    behavior = f"LUMPED into giant community (ID {c_idx}) containing {c_size} proteins"
                else:
                    behavior = f"CLEANLY RECOVERED in community (ID {c_idx})"
            else:
                pct = (overlap / len(prots)) * 100.0
                if c_size > len(prots) * 3:
                    behavior = f"PARTIALLY LUMPED ({overlap}/{len(prots)} subunits in giant ID {c_idx} of size {c_size})"
                else:
                    behavior = f"PARTIALLY RECOVERED ({overlap}/{len(prots)} subunits in ID {c_idx} of size {c_size})"
                    
            w(f"  {algo_name:<28} {overlap:>2}/{len(prots):<13} {c_size:>14}  {behavior}")
            
        w("")
        
    return "\n".join(lines)

def main():
    G = load_ppi_graph()
    cyc_complexes = load_cyc2008()
    
    # Run algorithms to get communities
    results = run_algorithms(G)
    
    # Phase 2: Pairwise retention rates
    df_retention = calculate_pairwise_retention(results, G)
    
    # Export CSV for Phase 2
    csv_path = "pairwise_retention.csv"
    df_retention.to_csv(csv_path, index=False)
    print(f"\n[OK] Phase 2: Pairwise retention table saved to '{csv_path}'")
    
    # Phase 3: Coarse community maps
    phase3_text = generate_phase3_reports(results, cyc_complexes)
    
    # Combine reports
    report_path = "pairwise_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==============================================================================\n")
        f.write("PHASE 2 & 3 EXTRANEOUS BIOLOGICAL REPORT (Jamilene's Completed Tasks)\n")
        f.write("==============================================================================\n\n")
        
        f.write("------------------------------------------------------------------------------\n")
        f.write("PHASE 2: PAIRWISE RETENTION RATES\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write("Shows how well each algorithm retains the fine-grained relationship pairs\n")
        f.write("from Dr. Villar's MCL baseline clusters across different inflation levels (I).\n\n")
        
        # Write retention table as text
        f.write(df_retention.to_string(index=False))
        f.write("\n\n")
        
        # Write phase 3 text
        f.write(phase3_text)
        
    print(f"[OK] Phase 3: Details & Report saved to '{report_path}'")
    print("\nAll tasks in Jamilene's list have been successfully executed!")

if __name__ == "__main__":
    main()
