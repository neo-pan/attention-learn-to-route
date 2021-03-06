import os
import os.path as osp
import pickle
from typing import List, Union

import networkx as nx
import networkx.algorithms.tree.mst as mst
import torch
from problems.tsp.state_tsp import StateTSP
from scipy.spatial.distance import cdist
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import Distance
from torch_geometric.utils import (from_networkx, to_dense_batch, to_networkx,
                                   to_undirected)
from utils.beam_search import beam_search

_curdir = osp.dirname(osp.abspath(__file__))
_fake_dataset_root = osp.join(_curdir, "FAKEDataset")
_preloaded_data = {}


class TSP(object):

    NAME = "tsp"

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi)
            == pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        if isinstance(dataset, Batch):
            dataset, _mask = to_dense_batch(dataset.pos, dataset.batch)
            assert (
                _mask.all()
            ), f"Only support batch of graphs with the same node numbers."
        elif isinstance(dataset, torch.Tensor):
            dataset = dataset
            assert dataset.dim() == 3
        else:
            raise TypeError(f"Unsupported data type {type(dataset)}")

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - d[:, -1]).norm(p=2, dim=1),
            None,
        )

    @staticmethod
    def make_dataset(*args, **kwargs):
        model = kwargs.pop("model")
        if model in ["gnn", "gnnff"]:
            return _GNN_TSPDataset(*args, **kwargs)
        elif model in ["attention", "pointer"]:
            return _TSPDataset(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported model type {model}")

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input,
        beam_size,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def gen_fully_connected_graph(
    data: Union[int, List[float]], pos_feature: bool = True
) -> Data:
    node_num = None
    node_pos = None
    if isinstance(data, int):
        node_num = data
    elif isinstance(data, list):
        node_pos = data
        node_num = len(node_pos)

    index = torch.arange(node_num).unsqueeze(-1).expand([-1, node_num])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    pos = (
        torch.tensor(node_pos)
        if node_pos
        else torch.empty(size=(node_num, 2)).uniform_(0, 1)
    )
    node_feat = torch.tensor(
        [[0, 1] if i == 0 else [1, 0] for i in range(node_num)], dtype=torch.float,
    )

    graph = Data(x=torch.cat([node_feat, pos], dim=-1), edge_index=edge_index, pos=pos)
    graph = Distance(norm=False, cat=False)(graph)

    return graph


def gen_mst_graph(data: Union[int, List[float]], pos_feature: bool = True) -> Data:
    graph = gen_fully_connected_graph(data, pos_feature)
    graph = Distance(norm=False, cat=False)(graph)
    nx_g = to_networkx(
        graph, node_attrs=["x", "pos"], edge_attrs=["edge_attr"], to_undirected=True
    )
    nx_mst = mst.minimum_spanning_tree(nx_g, weight="edge_attr", algorithm="prim")

    mst_graph = from_networkx(nx_mst)
    # mst_graph.x = graph.x
    # mst_graph.pos = graph.pos
    if mst_graph.edge_attr.ndim == 1:
        mst_graph.edge_attr.unsqueeze_(-1)

    return mst_graph


def gen_knn_graph(
    data: Union[int, List[float]], k=5, pos_feature: bool = True
) -> Data:
    graph = gen_fully_connected_graph(data, pos_feature)
    edge_index = knn_graph(graph.pos, k=k, loop=False)
    edge_index = to_undirected(edge_index)
    graph.edge_index = edge_index
    graph = Distance(norm=False, cat=False)(graph)

    return graph


def _gen_knn(dist, k):
    g_size = dist.shape[0]
    k_nn = dist.argsort()[:, 1 : k + 1]
    edge_list = []
    for u in range(g_size):
        for v in k_nn[u]:
            edge_list.append([u, v, {"is_mst": 0.0, "edge_attr": dist[u, v]}])
    g = nx.Graph(edge_list)

    return g


def gen_knn_mst_graph(
    data: Union[int, List[float]], k=5, pos_feature: bool = True
) -> Data:
    graph = gen_fully_connected_graph(data, pos_feature)
    graph = Distance(norm=False, cat=False)(graph)
    pos = graph.pos.numpy()
    dist = cdist(pos, pos, metric="euclidean")
    knn_g = _gen_knn(dist, k)
    nx_g = to_networkx(
        graph, node_attrs=["x", "pos"], edge_attrs=["edge_attr"], to_undirected=True
    )
    nx_mst = mst.minimum_spanning_tree(nx_g, weight="edge_attr", algorithm="prim")
    nx.set_edge_attributes(nx_mst, 1.0, "is_mst")

    knn_g.update(nx_mst)
    pyg = from_networkx(knn_g)
    if pyg.is_mst.ndim == 1:
        pyg.is_mst.unsqueeze_(-1)
    if pyg.edge_attr.ndim == 1:
        pyg.edge_attr.unsqueeze_(-1)

    return pyg


gen_methods = {
    "complete": gen_fully_connected_graph,
    "mst": gen_mst_graph,
    "knn": gen_knn_graph,
    "knn_mst": gen_knn_mst_graph,
}


class _GNN_TSPDataset(InMemoryDataset):
    def __init__(
        self,
        filename=None,
        size=50,
        num_samples=100000,
        offset=0,
        distribution="complete",
    ):

        gen_graph = gen_methods.get(distribution, None)
        assert gen_graph, f"Unsupport graph distribution {distribution}"

        if filename is not None:
            global _preloaded_data
            data = _preloaded_data.get(filename, None)
            if data is None:
                assert os.path.splitext(filename)[1] == ".pkl"
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    _preloaded_data[filename] = data
            if isinstance(data[0], list):
                self.__data_list__ = [gen_graph(row) for row in data]
            elif isinstance(data[0], Data):
                indices = torch.randperm(len(data))[:num_samples]
                self.__data_list__ = [data[idx] for idx in indices]
        else:
            # Sample points randomly in [0, 1] square
            self.__data_list__ = [gen_graph(size) for i in range(num_samples)]

        self.size = len(self.__data_list__)
        super(InMemoryDataset, self).__init__(root=_fake_dataset_root)

    @property
    def raw_file_names(self):
        return ["raw_files.pt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_classes(self):
        return 1

    def process(self):
        data_list = self.__data_list__
        self.data, self.slices = self.collate(data_list)

    def _process(self):
        super()._process()


class _TSPDataset(InMemoryDataset):
    def __init__(
        self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None
    ):
        super(_TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    torch.FloatTensor(row)
                    for row in (data[offset : offset + num_samples])
                ]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
