# Cora Graph Data Exploration and Conv1d Exercise

A completed educational notebook based on a course exercise. The original notebook says **“adapted from: Longa”**
The notebook loads the Cora citation graph with PyTorch Geometric, inspects node labels, features and masks, selects high-degree nodes, builds and relabels a subgraph, extracts a two-hop neighborhood, and demonstrates a pointwise `Conv1d` transformation and MSE gradient. The incomplete `TODO` cells and device-selection bug in the supplied notebook have been filled.

**Technical distinction:** `Conv1d(kernel_size=1)` acts on each node's three selected features independently. The demonstration does **not** aggregate neighbor messages, train a graph neural network classifier, or report node classification accuracy. Its `edge_index` is used for graph exploration, not inside the Conv1d forward pass.

## Run

Create an environment compatible with your PyTorch installation, then install `requirements.txt` and run the notebook from top to bottom:

```bash
python -m venv .venv
# Activate the virtual environment
pip install -r requirements.txt
jupyter notebook cora_graph_exercise_completed.ipynb
```

The first run downloads the Cora Planetoid dataset into `data/Cora/`; an internet connection is required. PyTorch Geometric platform-specific installations may require following the [official installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). The dataset folder is excluded from Git.

## Attribution and limitations

This is a completed learning exercise, not an original GNN research project. The exact Coursera course and instructor are not identified in the attached notebook, so this README does not invent them. If the course materials have sharing restrictions, publish this independent summary or a newly written demonstration rather than the course prompt. No model accuracy result is claimed.
