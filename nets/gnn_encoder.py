import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn.modules import pooling
from torch_geometric.data.batch import Batch
from torch_geometric.nn import (BatchNorm, InstanceNorm,  # TransformerConv,
                                global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax, to_dense_batch

pooling_funcs = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class GNNLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, normalization, concat=True, feed_forward_hidden=512):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert self.embed_dim % self.n_heads == 0
        if normalization == "batch":
            norm_class = BatchNorm
        else:
            raise ValueError(f"Unsupported Normalization method: {normalization}")
        self.concat = concat
        if self.concat:
            self.gnn = TransformerConv(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim // self.n_heads,
                heads=self.n_heads,
                concat=self.concat,
                edge_dim=self.embed_dim,
            )
        else:
            self.gnn = TransformerConv(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                heads=self.n_heads,
                concat=self.concat,
                edge_dim=self.embed_dim,
            )
        self.norm = norm_class(in_channels=self.embed_dim)
        self.feed_foward = nn.Sequential(
            nn.Linear(self.embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, self.embed_dim),
            norm_class(in_channels=self.embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        prev_alpha: torch.Tensor = None,
    ) -> torch.Tensor:
        x, alpha = self.gnn(x, edge_index, edge_attr, prev_alpha)
        x = self.norm(x)
        x = self.feed_foward(x)

        return x, alpha


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

        for _ in range(self.n_layers-1):
            gnn_layer = GNNLayer(self.n_heads, self.embed_dim, normalization, concat=True)
            gnn_layer_list.append(gnn_layer)
        # Output layer of encoder, use 'average' instead of 'concat'
        gnn_layer = GNNLayer(self.n_heads, self.embed_dim, normalization, concat=True)
        gnn_layer_list.append(gnn_layer)

        self.gnn_layer_list = nn.ModuleList(gnn_layer_list)

        # Add initial attention weights
        # self.init_att = nn.Sequential(
        #     nn.Linear(1, self.embed_dim),
        #     nn.Linear(self.embed_dim, self.n_heads)
        # )

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        edge_feat = data.edge_feat
        batch = data.batch
        alpha = data.is_mst.expand(-1, self.n_heads) if hasattr(data, "is_mst") else None
        for gnn_layer in self.gnn_layer_list:
            x, alpha = gnn_layer(x, edge_index, edge_feat, alpha)
            # x += out

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
        alpha = None
        for gnn, norm in zip(self.gnn_list, self.norm_list):
            x, alpha = gnn(x, edge_index, edge_feat, alpha)
            x = norm(x)

        dense_embeddings = to_dense_batch(x, batch)[0]
        graph_embeddings = self.pooling_func(x, batch)

        return (dense_embeddings, graph_embeddings)



class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (float, optional): If set, will combine aggregation and
            skip information via :math:`\beta\,\mathbf{W}_1 \vec{x}_i + (1 -
            \beta) \left(\sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2
            \vec{x}_j \right)`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: Optional[float] = None,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = Linear(1, 1)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        prev_alpha: OptTensor = None,
    ):

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        self._alpha = prev_alpha
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        del self._alpha

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        x = self.lin_skip(x[1])
        if self.beta is not None:
            out = self.beta * x + (1 - self.beta) * out
        else:
            out += x

        return out, alpha

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Optional[Tensor],
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:

        query = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        key = self.lin_query(x_i).view(-1, self.heads, self.out_channels)

        lin_edge = self.lin_edge
        if edge_attr is not None:
            edge_attr = lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key += edge_attr

        prev_alpha = self._alpha

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if prev_alpha is not None:
            assert prev_alpha.shape == alpha.shape
            alpha += prev_alpha
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        self._alpha = alpha

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

