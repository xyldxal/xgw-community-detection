import nbformat

with open('(PPI)_Network.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"--- Cell {i} ---")
        print(cell.source)
