import gzip
import json
from os import path as osp
import torch
from torch_geometric import data as tgdata


class MyDataset(tgdata.InMemoryDataset):
    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        self.json_gzip_path = f".dataset_src/den_graph_data_8_feature_{split}.json.gz"
        new_root = osp.join(root, split)
        super(MyDataset, self).__init__(
            new_root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        with gzip.open(self.json_gzip_path, "rt") as f:
            data_list_json = json.load(f)
            data_list = [
                tgdata.Data(
                    x=(
                        (lambda x: (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6) if x.numel() > 0 else x)
                        (torch.tensor(data["x"], dtype=torch.float32))
                    ),
                    edge_index=torch.tensor(
                        data["edge_index"], dtype=torch.long
                    ),
                    edge_attr=torch.tensor(
                        data["edge_attr"], dtype=torch.long
                    ),
                    y=torch.tensor(data["y"], dtype=torch.float32),
                )
                for data in data_list_json
            ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # Force x to be long dtype after preprocessing
        for data in data_list:
            if data.x is not None:
                data.x = data.x.long()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
   