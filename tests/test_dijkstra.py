from __future__ import annotations

import logging
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
import trimesh
from scipy.sparse import coo_matrix, csr_matrix

from rust_graph import all_pairs_dijkstra_path_length

logger = logging.getLogger(__name__)

TEST_DIR = Path(__file__).parent.parent / "data" / "tests"


def compare_dijkstra_rust_with_networkx(ply_path):
    radius = 12.0
    mesh: trimesh.Trimesh = trimesh.load(ply_path, force="mesh")  # type: ignore

    # Graph
    graph = nx.Graph()
    n = len(mesh.vertices)
    graph.add_nodes_from(np.arange(n))
    logger.info(f"{graph = }")

    # Get edges
    f = np.array(mesh.faces, dtype=int)
    logger.info(f"{f.shape = }")
    rowi = np.concatenate(
        [f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]], axis=0
    )
    rowj = np.concatenate(
        [f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]], axis=0
    )
    logger.info(f"{rowi.shape = }, {rowj.shape = }")
    verts = mesh.vertices
    logger.info(f"{verts.shape = }")

    # Get weights
    edgew = verts[rowi] - verts[rowj]
    edgew = scipy.linalg.norm(edgew, axis=1)
    wedges = np.stack([rowi, rowj, edgew]).T

    weighted_edges = list(zip(rowi, rowj, edgew))

    graph.add_weighted_edges_from(wedges)
    logger.info(graph)

    start = time.time()
    dists = nx.all_pairs_dijkstra_path_length(graph, cutoff=radius)
    d2 = {}

    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.time()
    logger.info(f"Dijkstra took {end - start:.2f} s")

    start = time.time()
    d3 = all_pairs_dijkstra_path_length(weighted_edges, cutoff=radius)
    end = time.time()
    logger.info(f"Rust Dijkstra took {end - start:.2f} s")

    # compare the two dictionaries (key: int, value: dict[int, float])
    assert d2.keys() == d3.keys(), f"{d2.keys() = }, {d3.keys() = }"
    for key in d2:
        assert d2[key].keys() == d3[key].keys()
        for key2 in d2[key]:
            assert d2[key][key2] == d3[key][key2]

    # sparse_d2 = dict_to_sparse(d2)
    # sparse_d3 = dict_to_sparse(d3)
    #
    # # PERF: comparing sparse matrices with == is slow, so we use !=
    #
    # # compare the two sparse matrices
    # assert (sparse_d2 != sparse_d3).nnz == 0


def dict_to_sparse(mydict):
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


def test_dijkstra_rust_with_networkx_1():
    compare_dijkstra_rust_with_networkx(TEST_DIR / "plys" / "2P1Q_B.ply")


def test_dijkstra_rust_with_networkx_2():
    compare_dijkstra_rust_with_networkx(TEST_DIR / "plys" / "2P1Q_C.ply")
