from collections import OrderedDict
from typing import List, Dict
import flwr as fl
import numpy as np
import torch

from flwr_datasets import FederatedDataset
from ray import client
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from src.modules.utils import train, test, apply_transforms
from src.modules.model import ModelFactory


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, config):
        self.trainset = trainset
        self.valset = valset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = ModelFactory.create_model(config).to(self.device)

    def get_parameters(self, config=None):
        """Retrieve model parameters as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def evaluate(self, parameters, config):
        """Evaluate the model using the validation dataset."""
        set_params(self.model, parameters)
        valloader = DataLoader(self.valset, batch_size=64)

        loss, accuracy = test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    # Placeholder fit method to be overridden by derived classes
    def fit(self, parameters, config):
        raise NotImplementedError("fit method must be implemented in subclasses.")


class FedAvgClient(FlowerClient):
    def fit(self, parameters, config):
        set_params(self.model, parameters)

        batch, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train(self.model, trainloader, optimizer, epochs=epochs, device=self.device)

        return self.get_parameters({}), len(trainloader.dataset), {}


class FedNovaClient(FlowerClient):
    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if self.exp_config.var_local_epochs:
            print('pot==================================')
            seed_val = (
                    2023
                    + int(self.client_id)
                    + int(self.exp_config.seed)
            )
            np.random.seed(seed_val)
            num_epochs = np.random.randint(
                self.exp_config.var_min_epochs, self.exp_config.var_max_epochs
            )
        else:
            num_epochs = self.num_epochs
        # Train
        train(self.model, trainloader, self.optimizer, epochs=num_epochs, device=self.device)

        # Get ratio by which the strategy would scale local gradients from each client
        # We use this scaling factor to aggregate the gradients on the server
        grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), grad_scaling_factor





def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_client_fn(dataset: FederatedDataset, num_classes):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(context) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(
            int(context.node_config["partition-id"]), "train"
        )

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainset, valset, num_classes).to_client()

    return client_fn

class ClientFactory:
    @staticmethod
    def get_client_fn(dataset: FederatedDataset, conf):
        """Return a function to construct a client.

        The VirtualClientEngine will execute this function whenever a client is sampled by
        the strategy to participate.
        """

        def client_fn(context) -> fl.client.Client:
            """Construct a FlowerClient with its own dataset partition."""

            # Let's get the partition corresponding to the i-th client
            client_dataset = dataset.load_partition(
                int(context.node_config["partition-id"]), "train"
            )

            # Now let's split it into train (90%) and validation (10%)
            client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

            trainset = client_dataset_splits["train"]
            valset = client_dataset_splits["test"]

            # Now we apply the transform to each batch.
            trainset = trainset.with_transform(apply_transforms)
            valset = valset.with_transform(apply_transforms)

            # Create and return client
            if conf.strategy.name == "FedAvg":
                return FedAvgClient(trainset, valset, conf).to_client()
            elif conf.strategy.name == "FedNova":
                return FedNovaClient(trainset, valset, conf).to_client()
            else:
                raise ValueError(f"Unsupported Algorithm name: {conf.model.name}")


        return client_fn