# ruff: noqa: T201 F841
from __future__ import annotations

import rich.traceback

rich.traceback.install(show_locals=True)

import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
import trimesh
from scipy.sparse import coo_matrix, csr_matrix

from rust_graph import all_pairs_dijkstra_path_length

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    ply_path_1 = DATA_DIR / "tests/plys/2P1Q_B.ply"
    ply_path_2 = DATA_DIR / "tests/plys/2P1Q_C.ply"
    radius = 12.0
    max_vertices = 200
    mesh: trimesh.Trimesh = trimesh.load(ply_path_1, force="mesh")  # type: ignore
    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals

    # Graph
    G = nx.Graph()  # noqa: N806
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))
    print(G)

    # Get edges
    f = np.array(mesh.faces, dtype=int)
    print(f.shape)
    rowi = np.concatenate(
        [f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]], axis=0
    )
    rowj = np.concatenate(
        [f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]], axis=0
    )
    print(rowi.shape, rowj.shape)
    verts = mesh.vertices
    print(verts, verts.shape)

    edgew = verts[rowi] - verts[rowj]
    edgew = scipy.linalg.norm(edgew, axis=1)
    wedges = np.stack([rowi, rowj, edgew]).T

    weighted_edges = list(zip(rowi, rowj, edgew))

    G.add_weighted_edges_from(wedges)
    print(G)

    start = time.time()
    dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius)
    d2 = {}

    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.time()
    print(f"networkx Dijkstra took {end - start:.2f} s")

    start = time.time()
    d3 = all_pairs_dijkstra_path_length(weighted_edges, cutoff=radius)
    end = time.time()
    print(f"rust-graph Dijkstra took {end - start:.2f} s")

    for start_node in d2:
        for end_node in d2[start_node]:
            if start_node == end_node:
                continue
            assert d2[start_node][end_node] == d3[start_node][end_node]

    sparse_d2 = dict_to_sparse(d2)
    sparse_d3 = dict_to_sparse(d3)

    # PERF: comparing sparse matrices with == is slow, so we use !=

    # compare the two sparse matrices
    assert (sparse_d2 != sparse_d3).nnz == 0


def dict_to_sparse(mydict: dict[int, dict[int, float]]):
    """Create a sparse matrix from a dictionary."""
    # Create the appropriate format for the COO format.
    data = []
    row = []
    col = []
    for r in mydict:
        for c in mydict[r]:
            r = int(r)
            c = int(c)
            v = mydict[r][c]
            data.append(v)
            row.append(r)
            col.append(c)
    # Create the COO-matrix
    coo = coo_matrix((data, (row, col)))
    # Let Scipy convert COO to CSR format and return
    return csr_matrix(coo)


if __name__ == "__main__":
    main()
