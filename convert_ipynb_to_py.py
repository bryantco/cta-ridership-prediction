import nbformat
from pathlib import Path

def nb_to_py(path):
    nb = nbformat.read(path, as_version=4)
    code = "\n\n".join(
        cell.source
        for cell in nb.cells
        if cell.cell_type == "code"
    )
    with open(path.replace(".ipynb", ".py"), "w") as f:
        f.write(code)

for nb_path in Path(".").glob("**/*.ipynb"):
    nb_to_py(str(nb_path))