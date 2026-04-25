import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell("# Community Detection on Yeast PPI Network\\n\\nThis notebook handles data loading, algorithm execution, ground-truth loading, and F-score evaluation."))

nb.cells.append(new_code_cell("""\\
!pip install --no-cache-dir cdlib leidenalg python-louvain networkx pandas scikit-learn matplotlib
"""))

nb.cells.append(new_code_cell("""\\
import networkx as nx
import pandas as pd
from cdlib import algorithms, evaluation, NodeClustering
import os
import numpy as np
from collections import Counter
from itertools import combinations

# 1. Ensure the PPI graph loads correctly from dip_ppin.csv
data_path = "xgw/data/preprocessed/dip_ppin.csv"
print(f"Loading PPI network from {data_path}...")
df = pd.read_csv(data_path)
cols = df.columns
source_col, target_col = cols[0], cols[1]
ppi_graph = nx.from_pandas_edgelist(df, source=source_col, target=target_col)

print(f"Nodes: {len(ppi_graph.nodes)}")
print(f"Edges: {len(ppi_graph.edges)}")
"""))

nb.cells.append(new_code_cell("""\\
# 2. Run all community detection algorithms
methods = {}

print("Running Leiden...")
methods['Leiden'] = algorithms.leiden(ppi_graph)
print(f"  Found {len(methods['Leiden'].communities)} groups")

print("Running Louvain...")
methods['Louvain'] = algorithms.louvain(ppi_graph)
print(f"  Found {len(methods['Louvain'].communities)} groups")

print("Running Label Propagation...")
methods['Label Propagation'] = algorithms.label_propagation(ppi_graph)
print(f"  Found {len(methods['Label Propagation'].communities)} groups")

print("Running Clique Percolation method (k=3)...")
methods['Clique Percolation'] = algorithms.kclique(ppi_graph, k=3)
print(f"  Found {len(methods['Clique Percolation'].communities)} groups")

print("Running Girvan-Newman...")
print("  (Skipped due to computational intensity on ~5k nodes/22k edges)")
# methods['Girvan-Newman'] = algorithms.girvan_newman(ppi_graph, level=1)

print("Running Stochastic Block Model...")
try:
    methods['SBM'] = algorithms.sbm_dl(ppi_graph)
    print(f"  Found {len(methods['SBM'].communities)} groups")
except Exception as e:
    print("  SBM failed (graph-tool likely missing). Skipping SBM.")

print("Running Consensus clustering algorithm...")
# Optimized Consensus: Count co-occurrences using Counter
fast_comms = [methods['Leiden'].communities, methods['Louvain'].communities, methods['Label Propagation'].communities]
co_occurrence = Counter()

for comms in fast_comms:
    for c in comms:
        # Sort to ensure consistent edge keys
        c_list = sorted(list(c))
        co_occurrence.update(combinations(c_list, 2))

threshold = len(fast_comms) / 2.0
filtered_edges = [(u, v, count) for (u, v), count in co_occurrence.items() if count > threshold]

filtered_G = nx.Graph()
filtered_G.add_nodes_from(ppi_graph.nodes())
filtered_G.add_weighted_edges_from(filtered_edges)

consensus_clusters = algorithms.louvain(filtered_G)
methods['Consensus'] = NodeClustering(consensus_clusters.communities, ppi_graph, "Consensus")
print(f"  Found {len(methods['Consensus'].communities)} groups")
"""))

nb.cells.append(new_code_cell("""\\
# 3. Load MCL reference clusters and CYC2008 ground-truth complexes
ground_truths = {}

def load_mcl_clusters(filepath, graph):
    clusters = []
    with open(filepath, 'r') as f:
        for line in f:
            members = line.strip().split('\\t')
            valid_members = [m for m in members if m in graph]
            if len(valid_members) > 0:
                clusters.append(valid_members)
    return NodeClustering(clusters, graph, "MCL")

mcl_levels = [20, 30, 40, 50]
for lvl in mcl_levels:
    mcl_path = f"xgw/data/clusters/out.dip_unweighted.csv.I{lvl}"
    ground_truths[f'MCL I={lvl/10.0}'] = load_mcl_clusters(mcl_path, ppi_graph)
    print(f"MCL I={lvl/10.0} found {len(ground_truths[f'MCL I={lvl/10.0}'].communities)} groups")

cyc_path = "xgw/data/swc/complexes_CYC.txt"
cyc_df = pd.read_csv(cyc_path, sep='\\t', header=None, names=['protein', 'complex_id', 'complex_name'])
cyc_clusters_raw = [group['protein'].tolist() for _, group in cyc_df.groupby('complex_id')]
cyc_clusters = [[m for m in c if m in ppi_graph] for c in cyc_clusters_raw]
cyc_clusters = [c for c in cyc_clusters if len(c) > 0]
ground_truths['CYC2008'] = NodeClustering(cyc_clusters, ppi_graph, "CYC2008")
print(f"CYC2008 found {len(ground_truths['CYC2008'].communities)} groups")
"""))

nb.cells.append(new_code_cell("""\\
# 4. Implement F-score matching evaluation
print("\\n--- F-score Matching Evaluation ---")
for gt_name, gt_clustering in ground_truths.items():
    print(f"\\nGround Truth: {gt_name}")
    for algo_name, algo_clustering in methods.items():
        try:
            f_score = evaluation.f1(algo_clustering, gt_clustering).score
            print(f"  {algo_name:20s}: {f_score:.4f}")
        except Exception as e:
            print(f"  {algo_name:20s}: Failed ({e})")
"""))

nb.cells.append(new_code_cell("""\\
# 5. Export GEXF files for Gephi with Python
import xml.etree.ElementTree as ET

os.makedirs('gephi_exports', exist_ok=True)

print("Computing spring layout for Gephi exports (iterations=20)...")
pos = nx.spring_layout(ppi_graph, k=0.15, iterations=20, seed=42)
scale = 1000
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}
print("Layout computed.")

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

for algo_name, clustering in methods.items():
    G = ppi_graph.copy()
    safe_name = algo_name.lower().replace(' ', '_').replace('-', '_')
    
    for comm_id, members in enumerate(clustering.communities):
        for node in members:
            if node in G.nodes:
                G.nodes[node]['community'] = comm_id
                
    for node in G.nodes:
        if 'community' not in G.nodes[node]:
            G.nodes[node]['community'] = -1
            
    out_path = f'gephi_exports/ppi_{safe_name}.gexf'
    nx.write_gexf(G, out_path)
    inject_positions_into_gexf(out_path, pos)
    print(f"Exported: {out_path}")

print("\\nAll exports complete! Files are ready in the gephi_exports/ folder.")
"""))

with open('(PPI)_Network.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook (PPI)_Network.ipynb successfully generated!")
