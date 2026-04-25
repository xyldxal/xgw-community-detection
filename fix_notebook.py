import nbformat

with open('(PPI)_Network.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Look for cell containing pip install
for cell in nb.cells:
    if cell.cell_type == 'code' and 'pip install' in cell.source:
        cell.source = "!pip install --no-cache-dir cdlib leidenalg python-louvain networkx pandas scikit-learn matplotlib"

with open('(PPI)_Network.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook pip cell fixed.")
