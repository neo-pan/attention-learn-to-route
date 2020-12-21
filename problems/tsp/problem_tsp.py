import os
import os.path as osp
import pickle

import torch
from problems.tsp.state_tsp import StateTSP
from torch_geometric.data import Data, InMemoryDataset
from utils.beam_search import beam_search
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from typing import List

_curdir = osp.dirname(osp.abspath(__file__))
_fake_dataset_root = osp.join(_curdir, "FAKEDataset")

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
            assert _mask.all(), f"Only support batch of graphs with the same node numbers."
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
        if model == "gnn":
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


def gen_fully_connected_graph(node_num: int, pos_feature: bool = True) -> Data:
    index = torch.arange(node_num).unsqueeze(-1).expand([-1, node_num])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    pos = torch.empty(size=(node_num, 2)).uniform_(0, 1)
    node_feat = torch.tensor(
        [[0, 1] if i == 0 else [1, 0] for i in range(node_num)], dtype=torch.float,
    )

    graph = Data(x=torch.cat([node_feat, pos], dim=-1), edge_index=edge_index, pos=pos)

    return graph

def gen_fully_connected_graph_with_pos(node_pos: List[float], pos_feature: bool = True) -> Data:
    node_num = len(node_pos)
    index = torch.arange(node_num).unsqueeze(-1).expand([-1, node_num])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    pos = torch.tensor(node_pos)
    node_feat = torch.tensor(
        [[0, 1] if i == 0 else [1, 0] for i in range(node_num)], dtype=torch.float,
    )

    graph = Data(x=torch.cat([node_feat, pos], dim=-1), edge_index=edge_index, pos=pos)

    return graph


class _GNN_TSPDataset(InMemoryDataset):
    def __init__(
        self,
        filename=None,
        size=50,
        num_samples=100000,
        offset=0,
        distribution="complete",
    ):

        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.__data_list__ = [
                    gen_fully_connected_graph_with_pos(row)
                    for row in data
                ]
        else:
            # Sample points randomly in [0, 1] square
            self.__data_list__ = [
                gen_fully_connected_graph(node_num=size) for i in range(num_samples)
            ]

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
