# rust-graph: Dijkstra written in Rust

[![image](https://img.shields.io/pypi/v/rust-graph.svg)](https://pypi.python.org/pypi/rust-graph)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rust-graph)](https://pypistats.org/packages/rust-graph)
[![image](https://img.shields.io/pypi/l/rust-graph.svg)](https://pypi.python.org/pypi/rust-graph)
[![image](https://img.shields.io/pypi/pyversions/rust-graph.svg)](https://pypi.python.org/pypi/rust-graph)

|  |  |
|--|--|
|[![Ruff](https://img.shields.io/badge/Ruff-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/astral-sh/ruff) [![rustfmt](https://img.shields.io/badge/rustfmt-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://github.com/rust-lang/rustfmt) |[![Actions status](https://github.com/deargen/rust-graph/workflows/Style%20checking/badge.svg)](https://github.com/deargen/rust-graph/actions)|
| [![Ruff](https://img.shields.io/badge/Ruff-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/astral-sh/ruff) [![Clippy](https://img.shields.io/badge/clippy-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://github.com/rust-lang/rust-clippy) | [![Actions status](https://github.com/deargen/rust-graph/workflows/Linting/badge.svg)](https://github.com/deargen/rust-graph/actions) |
| [![pytest](https://img.shields.io/badge/pytest-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/pytest-dev/pytest) [![cargo test](https://img.shields.io/badge/cargo%20test-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://doc.rust-lang.org/cargo/commands/cargo-test.html) | [![Actions status](https://github.com/deargen/rust-graph/workflows/Tests/badge.svg)](https://github.com/deargen/rust-graph/actions) |
| [![uv](https://img.shields.io/badge/uv-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/astral-sh/uv) | [![Actions status](https://github.com/deargen/rust-graph/workflows/Check%20pip%20compile%20sync/badge.svg)](https://github.com/deargen/rust-graph/actions) |


Graph algorithms implemented in Rust, available as a Python package. >10x faster than `networkx`.

So far, there is only one function implemented: `all_pairs_dijkstra_path_length`. It's a re-write of the `networkx` function with the same name and should return the same results.

## 🛠️ Installation

```bash
pip install rust-graph
```

## 🚦 Usage

```python
from rust_graph import all_pairs_dijkstra_path_length

weighted_edges = [
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 3.0),
    (3, 0, 4.0),
    (0, 3, 5.0),
]

shortest_paths = all_pairs_dijkstra_path_length(weighted_edges, cutoff=3.0)
```

```python
>>> shortest_paths
{3: {3: 0.0, 2: 3.0}, 2: {2: 0.0, 1: 2.0, 0: 3.0, 3: 3.0}, 1: {0: 1.0, 2: 2.0, 1: 0.0}, 0: {1: 1.0, 0: 0.0, 2: 3.0}}
```

## 📈 Benchmark

Tried a couple of options but failed for various reasons. Here are some notes on them:

- [cugraph](https://developer.nvidia.com/blog/accelerating-networkx-on-nvidia-gpus-for-high-performance-graph-analytics/): 
    - Slower than `networkx` for the test data.
    - Not available on PyPI, only supports python 3.10 (and not above) and some dependencies were broken, making it difficult to set up.
- [rustworkx](https://www.rustworkx.org/): 
    - `cutoff` parameter is not implemented.
    - Extremely slow when the output is too large, because it returns lazy types rather than the actual values and converting it is probably not memory efficient.


Thus, we compare the performance of `networkx` and `rust-graph` for the `all_pairs_dijkstra_path_length` function.


### MacBook Pro (M1)

23x as fast as `networkx`:

```
networkx Dijkstra took 4.45 s
rust-graph Dijkstra took 0.19 s
```


### Personal laptop (AMD Ryzen R7 5800H (8 cores, 20MB total cache, 3.2 GHz, boost up to 4.4 GHz))

12x as fast as `networkx`:

```
networkx Dijkstra took 6.83 s
rust-graph Dijkstra took 0.57 s
```

If not using rayon parallelism, it's twice as slow:

```
networkx Dijkstra took 7.12 s
rust-graph Dijkstra took 1.04 s
```

### Azure server (AMD EPYC 7V13 64-Core Processor)

CPU info:

```
    Model name:            AMD EPYC 7V13 64-Core Processor
    CPU family:          25
    Model:               1
    Thread(s) per core:  1
    Core(s) per socket:  48
```

15x as fast as `networkx`:

```
networkx Dijkstra took 6.14 s
rust-graph Dijkstra took 0.41 s
```

## 👨‍💻️ Maintenance Notes

### Install from source

Install `uv`, `rustup` and `maturin`. Activate a virtual environment. Then,

```bash
bash scripts/install.sh
uv pip install -r deps/requirements_dev.in
python3 scripts/hf_download.py  # Download test data
```

### Run benchmarks

```bash
python3 tools/benchmark.py
```

### Compile requirements (generate lockfiles)

Use GitHub Actions: `apply-pip-compile.yml`. Manually launch the workflow and it will make a commit with the updated lockfiles.

