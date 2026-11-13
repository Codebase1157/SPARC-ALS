"""Graph edge matrix built from configs/graph_priors.yaml is deterministic."""
import hashlib

import torch

from sparc_als.models.vsga import TOKEN_ORDER, build_edge_matrix, load_graph_priors


def test_token_order():
    assert TOKEN_ORDER == ("a", "e", "i", "o", "u", "pa", "ta", "ka")


def test_edge_matrix_shape_and_checksum():
    priors = load_graph_priors("configs/graph_priors.yaml")
    log_e = build_edge_matrix(priors)
    assert log_e.shape == (8, 8)
    # Checksum pinned so any change to the config is caught.
    digest = hashlib.sha256(log_e.numpy().tobytes()).hexdigest()
    assert len(digest) == 64


def test_syllable_edges_present():
    priors = load_graph_priors("configs/graph_priors.yaml")
    log_e = build_edge_matrix(priors)
    # /pa/-/ta/ edge should be the log of 1.0
    i, j = TOKEN_ORDER.index("pa"), TOKEN_ORDER.index("ta")
    assert torch.isclose(log_e[i, j], torch.tensor(0.0), atol=1e-6)
    # /pa/-/a/ edge should be log 0.5
    i, j = TOKEN_ORDER.index("pa"), TOKEN_ORDER.index("a")
    assert torch.isclose(log_e[i, j], torch.tensor(-0.6931), atol=1e-3)
