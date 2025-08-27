import gzip
import json
from os import path as osp
import torch
from torch_geometric import data as tgdata
import numpy as np

class MyDataset(tgdata.InMemoryDataset):
    def __init__(
        self, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        self.json_gzip_path = f".dataset_src/den_graph_data_7_{split}.json.gz"
        self.feature_dir = ".dataset_src/features_loop_7"
        self.feature_names = [
            "closeness",
            "degree",
            "betweenness",
            "clustering",
            "pagerank",
            "face_count",
        ]
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
        # Load JSON.gz
        with gzip.open(self.json_gzip_path, "rt") as f:
            data_list_json = json.load(f)

        # Load node features
        feature_arrays = {
            name: np.load(f"{self.feature_dir}/{name}.npy", allow_pickle=True)
            for name in self.feature_names
        }

        data_list = []
        for i, graph_data in enumerate(data_list_json):
            # Concatenate features for this graph
            feats = []
            for name in self.feature_names:
                f = feature_arrays[name][i]  # (num_nodes,) or (num_nodes, 1)
                if f.ndim == 1:
                    f = np.expand_dims(f, -1)
                feats.append(f)
            node_feat = np.concatenate(feats, axis=1)  # shape: (num_nodes, num_features)

            # Construct PyG Data object
            edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
            edge_attr = (
                torch.tensor(graph_data["edge_attr"], dtype=torch.float)
                if "edge_attr" in graph_data
                else torch.zeros(edge_index.shape[1], dtype=torch.float)
            )

            data_list.append(
                tgdata.Data(
                    x=torch.tensor(node_feat, dtype=torch.float),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(graph_data["y"], dtype=torch.float),
                )
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✅ Processed {len(data_list)} graphs with float node features")
