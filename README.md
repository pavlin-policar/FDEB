# Force-Directed Edge Bundling (FDEB)

[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

`fdeb` can be easily installed through pip using

```
pip install fdeb
```

## Usage

```python
edges = ...  # (N, 2, 2) -> N edges, each with two endpoints in 2D

from fdeb import fdeb
optimized_edges = fdeb(edges)
```

## Notes

This package currently implements a numpy-only version of the FDEB algorithm, which has asymptotic complexity O(n^2), making it too slow and memory-hungry for large numbers of graphs.

A numba version of this algorithm is also available at https://github.com/verasativa/python.ForceBundle, however, it also implements the O(n^2) algorithm, so it should (asymptotically) run equally slowly as this package, but is more memory efficient, and perhaps a bit faster on larger graphs.
