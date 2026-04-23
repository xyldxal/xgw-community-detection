"""
Community Detection on Yeast PPI Network
=========================================
Copy each section (separated by # %% markers) into separate notebook cells,
or run this file directly: python community_detection.py
"""

# %% Cell 1 — Imports & Load PPI Graph
import networkx as nx
import pandas as pd
from cdlib import algorithms
import os

data_path = "xgw/data/databases/Scere20170205.txt"
df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='#', usecols=[0, 1])
ppi_graph = nx.from_pandas_edgelist(df, source=0, target=1)

print(f"Nodes: {len(ppi_graph.nodes)}")
print(f"Edges: {len(ppi_graph.edges)}")

# %% Cell 2 — Leiden & Louvain
print("\nRunning Leiden...")
leiden_clusters = algorithms.leiden(ppi_graph)
print(f"Leiden found {len(leiden_clusters.communities)} groups")

print("Running Louvain...")
louvain_clusters = algorithms.louvain(ppi_graph)
print(f"Louvain found {len(louvain_clusters.communities)} groups")

# %% Cell 3 — Label Propagation
# LP tends to create many tiny communities; we filter out groups smaller than MIN_SIZE.
MIN_COMMUNITY_SIZE = 3

print("\nRunning Label Propagation...")
lp_clusters_raw = algorithms.label_propagation(ppi_graph)
lp_communities = [c for c in lp_clusters_raw.communities if len(c) >= MIN_COMMUNITY_SIZE]
print(f"Label Propagation: {len(lp_clusters_raw.communities)} raw → {len(lp_communities)} after filtering (min size={MIN_COMMUNITY_SIZE})")

# %% Cell 5 — Load MCL reference clusters & CYC2008 ground truth
def load_mcl_clusters(filepath):
    """Load MCL cluster file: each line is a tab-separated cluster of protein IDs."""
    clusters = []
    with open(filepath, 'r') as f:
        for line in f:
            members = line.strip().split('\t')
            if members and members[0]:
                clusters.append(members)
    return clusters

# Lower inflation = fewer, larger clusters (I=2.0 is the coarsest available)
mcl_path = "xgw/data/clusters/out.dip_unweighted.csv.I20"
mcl_clusters_raw = load_mcl_clusters(mcl_path)
mcl_clusters = [c for c in mcl_clusters_raw if len(c) >= MIN_COMMUNITY_SIZE]
print(f"\nMCL (DIP unweighted, I=2.0): {len(mcl_clusters_raw)} raw → {len(mcl_clusters)} after filtering (min size={MIN_COMMUNITY_SIZE})")

# CYC2008 ground-truth protein complexes
cyc_path = "xgw/data/swc/complexes_CYC.txt"
cyc_df = pd.read_csv(cyc_path, sep='\t', header=None, names=['protein', 'complex_id', 'complex_name'])
cyc_clusters = [group['protein'].tolist() for _, group in cyc_df.groupby('complex_id')]
print(f"CYC2008 ground-truth complexes: {len(cyc_clusters)}")

for name, cl in [('MCL', mcl_clusters), ('CYC2008', cyc_clusters)]:
    sizes = [len(c) for c in cl]
    print(f"  {name}: sizes range {min(sizes)}-{max(sizes)}, median={sorted(sizes)[len(sizes)//2]}")

# %% Cell 6 — Pairwise NMI and ARI comparison
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def communities_to_labels(communities, all_nodes):
    """Convert list-of-lists communities to a label array aligned with all_nodes."""
    node_to_label = {}
    for cid, members in enumerate(communities):
        for node in members:
            node_to_label[node] = cid
    noise_label = len(communities)
    return [node_to_label.get(n, noise_label) for n in all_nodes]

all_nodes = sorted(ppi_graph.nodes())

methods = {
    'Leiden': leiden_clusters.communities,
    'Louvain': louvain_clusters.communities,
    'Label Propagation': lp_communities,
    'MCL (DIP, I=2.0)': mcl_clusters,
}

label_arrays = {name: communities_to_labels(comms, all_nodes) for name, comms in methods.items()}

method_names = list(methods.keys())
n = len(method_names)
nmi_matrix = np.zeros((n, n))
ari_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        nmi_matrix[i, j] = normalized_mutual_info_score(
            label_arrays[method_names[i]], label_arrays[method_names[j]]
        )
        ari_matrix[i, j] = adjusted_rand_score(
            label_arrays[method_names[i]], label_arrays[method_names[j]]
        )

nmi_df = pd.DataFrame(nmi_matrix, index=method_names, columns=method_names)
ari_df = pd.DataFrame(ari_matrix, index=method_names, columns=method_names)

print("\n=== Normalized Mutual Information (NMI) ===")
print(nmi_df.round(4).to_string())
print("\n=== Adjusted Rand Index (ARI) ===")
print(ari_df.round(4).to_string())

# %% Cell 7 — Community size distributions (plot + table)
import matplotlib.pyplot as plt

colors = {'Leiden': '#4C72B0', 'Louvain': '#DD8452',
          'Label Propagation': '#55A868', 'MCL (DIP, I=2.0)': '#C44E52'}

# --- Plot 1: Side-by-side box plot (log scale) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.5]})
fig.suptitle('Community Size Analysis', fontsize=15, fontweight='bold', y=1.02)

size_data = []
labels = []
box_colors = []
for name, comms in methods.items():
    sizes = [len(c) for c in comms]
    size_data.append(sizes)
    labels.append(f"{name}\n({len(comms)} groups)")
    box_colors.append(colors.get(name, '#999999'))

bp = ax1.boxplot(size_data, labels=labels, patch_artist=True, showfliers=True,
                 flierprops=dict(marker='.', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax1.set_yscale('log')
ax1.set_ylabel('Community size (log scale)')
ax1.set_title('Size Distribution Comparison', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# --- Plot 2: Top-15 largest communities per method ---
top_n = 15
for name, comms in methods.items():
    sizes = sorted([len(c) for c in comms], reverse=True)[:top_n]
    ax2.plot(range(1, len(sizes) + 1), sizes, 'o-', label=name,
             color=colors.get(name, '#999999'), markersize=5, linewidth=2, alpha=0.8)
ax2.set_xlabel('Community rank')
ax2.set_ylabel('Size (# proteins)')
ax2.set_title(f'Top {top_n} Largest Communities', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('community_size_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: community_size_distributions.png")

# Summary table
summary_data = []
for name, comms in methods.items():
    sizes = [len(c) for c in comms]
    summary_data.append({
        'Method': name,
        '# Communities': len(comms),
        'Min Size': min(sizes),
        'Max Size': max(sizes),
        'Mean Size': f"{np.mean(sizes):.1f}",
        'Median Size': int(np.median(sizes)),
        'Singletons': sum(1 for s in sizes if s == 1),
    })
summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# %% Cell 8 — Export GEXF files for Gephi (with pre-computed layout)
import xml.etree.ElementTree as ET

os.makedirs('gephi_exports', exist_ok=True)

# --- Pre-compute a force-directed layout once (reused for all exports) ---
print("Computing spring layout (this takes a moment for 5k nodes)...")
pos = nx.spring_layout(ppi_graph, k=0.15, iterations=80, seed=42)
# Scale positions so they spread well in Gephi (default is 0-1 range)
scale = 1000
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}
print("Layout computed.\n")


def inject_positions_into_gexf(filepath, positions):
    """Post-process a GEXF file to add <viz:position x='' y=''/> to each node."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Register namespaces so ET uses correct prefixes on write
    ET.register_namespace('', 'http://www.gexf.net/1.2draft')
    ET.register_namespace('viz', 'http://www.gexf.net/1.2draft/viz')

    # REMOVE explicit xmlns:viz attr — register_namespace already handles it.
    # Having both causes a duplicate declaration that Gephi's parser rejects.
    for attr in list(root.attrib):
        if 'viz' in attr and 'xmlns' in attr:
            del root.attrib[attr]

    # Find all <node> elements and inject <viz:position>
    for node_elem in root.iter('{http://www.gexf.net/1.2draft}node'):
        node_id = node_elem.get('id')
        if node_id in positions:
            x, y = positions[node_id]
            viz_pos = ET.SubElement(node_elem, '{http://www.gexf.net/1.2draft/viz}position')
            viz_pos.set('x', str(round(x, 2)))
            viz_pos.set('y', str(round(y, 2)))
            viz_pos.set('z', '0.0')

    tree.write(filepath, xml_declaration=True, encoding='utf-8')


# --- Individual GEXF files per method ---
for algo_name, comms in methods.items():
    G = ppi_graph.copy()
    safe_name = algo_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')

    for comm_id, members in enumerate(comms):
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

# --- Combined file with ALL community labels ---
G_all = ppi_graph.copy()
for algo_name, comms in methods.items():
    attr_name = algo_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
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
print(f"\nExported: {combined_path} (all community labels in one file)")
print("\nOpen in Gephi — layout is already applied!")
print("Go to Appearance → Nodes → Partition → pick any method to color by.")
