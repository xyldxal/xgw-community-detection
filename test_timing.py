import networkx as nx
import pandas as pd
from cdlib import algorithms, evaluation, NodeClustering
import os
import time

print("Loading PPI network...")
df = pd.read_csv("xgw/data/preprocessed/dip_ppin.csv")
ppi_graph = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
print(f"Nodes: {len(ppi_graph.nodes)}, Edges: {len(ppi_graph.edges)}")

methods = {}
t0 = time.time()
print("Running Leiden...")
methods['Leiden'] = algorithms.leiden(ppi_graph)

print(f"Running Louvain... {time.time()-t0:.2f}s")
methods['Louvain'] = algorithms.louvain(ppi_graph)

print(f"Running Label Propagation... {time.time()-t0:.2f}s")
methods['Label Propagation'] = algorithms.label_propagation(ppi_graph)

print(f"Running Clique Percolation... {time.time()-t0:.2f}s")
# If it's too slow we might skip it
# methods['Clique Percolation'] = algorithms.kclique(ppi_graph, k=3)

print(f"Running Girvan-Newman... {time.time()-t0:.2f}s")
# GN is extremely slow, maybe we skip it for now and see
# methods['Girvan-Newman'] = algorithms.girvan_newman(ppi_graph, level=1)

print(f"Running Consensus... {time.time()-t0:.2f}s")
consensus_G = nx.Graph()
consensus_G.add_nodes_from(ppi_graph.nodes())
fast_comms = [methods['Leiden'].communities, methods['Louvain'].communities, methods['Label Propagation'].communities]
for comms in fast_comms:
    for c in comms:
        c = list(c)
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                if consensus_G.has_edge(c[i], c[j]):
                    consensus_G[c[i]][c[j]]['weight'] += 1
                else:
                    consensus_G.add_edge(c[i], c[j], weight=1)

threshold = len(fast_comms) / 2.0
filtered_edges = [(u, v, d) for u, v, d in consensus_G.edges(data=True) if d['weight'] > threshold]
filtered_G = nx.Graph()
filtered_G.add_nodes_from(ppi_graph.nodes())
filtered_G.add_edges_from(filtered_edges)
consensus_clusters = algorithms.louvain(filtered_G)
methods['Consensus'] = NodeClustering(consensus_clusters.communities, ppi_graph, "Consensus")

print(f"Loading Ground Truths... {time.time()-t0:.2f}s")
ground_truths = {}
for lvl in [20, 30, 40, 50]:
    clusters = []
    with open(f"xgw/data/clusters/out.dip_unweighted.csv.I{lvl}", 'r') as f:
        for line in f:
            valid = [m for m in line.strip().split('\\t') if m in ppi_graph]
            if valid: clusters.append(valid)
    ground_truths[f'MCL I={lvl/10.0}'] = NodeClustering(clusters, ppi_graph, "MCL")

cyc_df = pd.read_csv("xgw/data/swc/complexes_CYC.txt", sep='\\t', header=None, names=['protein', 'complex_id', 'complex_name'])
cyc_clusters = [[m for m in group['protein'].tolist() if m in ppi_graph] for _, group in cyc_df.groupby('complex_id')]
cyc_clusters = [c for c in cyc_clusters if len(c) > 0]
ground_truths['CYC2008'] = NodeClustering(cyc_clusters, ppi_graph, "CYC2008")

print(f"Evaluation... {time.time()-t0:.2f}s")
for gt_name, gt in ground_truths.items():
    for name, algo in methods.items():
        print(f"{gt_name} vs {name}: {evaluation.f1(algo, gt).score:.4f}")

print(f"Done in {time.time()-t0:.2f}s")
