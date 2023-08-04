import torch
from torch_geometric.data import Data

class GNNTransformQM(object):
    def __call__(self, data_dict):
        # Extract node features (atoms) from the dictionary
        atom_features = data_dict["atoms"]
        num_atoms = len(atom_features)
        num_atom_features = len(atom_features.columns)

        # Extract edge indices (bonds) from the dictionary
        edge_indices = data_dict["bonds"]
        num_bonds = edge_indices.shape[0]

        # Extract additional feature (gfn0) from the dictionary
        gfn0_features = data_dict["gfn0"]
        num_gfn0_features = 1  # Assuming it's a single scalar value per atom (you can modify if needed)

        # Extract target labels from the dictionary (optional)
        target_labels = data_dict.get("labels", None)

        # Prepare node features (atom features)
        atom_features_tensor = torch.tensor(atom_features.values, dtype=torch.float)

        # Prepare edge indices (bonds)
        edge_indices_tensor = torch.tensor(edge_indices.T, dtype=torch.long)

        # Prepare additional feature (gfn0)
        gfn0_features_tensor = torch.tensor(gfn0_features, dtype=torch.float).view(-1, num_gfn0_features)

        # Prepare target labels (optional)
        if target_labels is not None:
            target_labels_tensor = torch.tensor(target_labels, dtype=torch.float)
        else:
            target_labels_tensor = None

        # Set node positions to an empty tensor
        pos_tensor = torch.empty((num_atoms, 0), dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(
            x=atom_features_tensor,
            edge_index=edge_indices_tensor,
            pos=pos_tensor,  # Set the node positions to an empty tensor
            gfn0=gfn0_features_tensor,  # Add the additional feature to the Data object
            y=target_labels_tensor,
            pid=data_dict["id"]
        )

        return data
