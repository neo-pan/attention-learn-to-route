import torch
import numpy as np
import torch.nn as nn
import math
from torch.nn.modules import pooling
from torch_geometric.data.batch import Batch
from typing import Tuple

from torch_geometric.nn import (
    TransformerConv,
    BatchNorm,
    InstanceNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

pooling_funcs = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        normalization="batch",
        pooling="mean",
        **kwarg,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert self.embed_dim % self.n_heads == 0
        self.n_layers = n_layers
        if normalization == "batch":
            norm_class = BatchNorm
        else:
            raise ValueError(f"Unsupported Normalization method: {normalization}")

        self.pooling_func = pooling_funcs.get(pooling, None)
        if self.pooling_func is None:
            raise ValueError(f"Unsupported Pooling method: {pooling}")

        gnn_list = []
        norm_list = []

        for _ in range(self.n_layers):
            gnn = TransformerConv(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim // self.n_heads,
                heads=self.n_heads,
            )
            norm = norm_class(in_channels=self.embed_dim)
            gnn_list.append(gnn)
            norm_list.append(norm)

        self.gnn_list = nn.ModuleList(gnn_list)
        self.norm_list = nn.ModuleList(norm_list)

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for gnn, norm in zip(self.gnn_list, self.norm_list):
            x = x + gnn(x, edge_index)
            x = norm(x)

        dense_embeddings = to_dense_batch(x, batch)[0]
        graph_embeddings = self.pooling_func(x, batch)

        return (dense_embeddings, graph_embeddings)
