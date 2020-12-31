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


class GNNLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, normalization, feed_forward_hidden=512):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert self.embed_dim % self.n_heads == 0
        if normalization == "batch":
            norm_class = BatchNorm
        else:
            raise ValueError(f"Unsupported Normalization method: {normalization}")

        self.gnn = TransformerConv(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim // self.n_heads,
            heads=self.n_heads,
        )
        self.norm = norm_class(in_channels=self.embed_dim)
        self.feed_foward = nn.Sequential(
            nn.Linear(self.embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, self.embed_dim),
            norm_class(in_channels=self.embed_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gnn(x, edge_index)
        x = self.norm(x)
        x = self.feed_foward(x)

        return x


class GraphTransformerFFEncoder(nn.Module):
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
        self.pooling_func = pooling_funcs.get(pooling, None)
        if self.pooling_func is None:
            raise ValueError(f"Unsupported Pooling method: {pooling}")

        gnn_layer_list = []

        for _ in range(self.n_layers):
            gnn_layer = GNNLayer(self.n_heads, self.embed_dim, normalization)
            gnn_layer_list.append(gnn_layer)

        self.gnn_layer_list = nn.ModuleList(gnn_layer_list)

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        for gnn_layer in self.gnn_layer_list:
            x = gnn_layer(x, edge_index)

        dense_embeddings = to_dense_batch(x, batch)[0]
        graph_embeddings = self.pooling_func(x, batch)

        return (dense_embeddings, graph_embeddings)


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
                edge_dim=self.embed_dim
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
        edge_feat = data.edge_feat

        for gnn, norm in zip(self.gnn_list, self.norm_list):
            x = x + gnn(x, edge_index, edge_feat)
            x = norm(x)

        dense_embeddings = to_dense_batch(x, batch)[0]
        graph_embeddings = self.pooling_func(x, batch)

        return (dense_embeddings, graph_embeddings)
