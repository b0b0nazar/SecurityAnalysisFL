from collections import OrderedDict
from typing import List, Tuple, Dict

import flwr as fl
import torch
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics
from flwr.common.typing import Scalar
from pyarrow import Scalar
from torch.utils.data import DataLoader

from src.modules.utils import test, apply_transforms
from src.modules.model import ModelFactory


def fit_config(epochs: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 2,  # Number of local epochs done by clients
        "batch_size": 64,  # Batch size to use by clients during fit()
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}





def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class ServerFactory:
    @staticmethod
    def get_evaluate_fn(centralized_testset: Dataset, conf):
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
                server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ):
            """Use the test set for evaluation."""

            # Determine device
            model = ModelFactory.create_model(conf)

            set_params(model, parameters)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Apply transform to dataset
            testset = centralized_testset.with_transform(apply_transforms)

            # Disable tqdm for dataset preprocessing
            disable_progress_bar()

            testloader = DataLoader(testset, batch_size=32)
            loss, accuracy = test(model, testloader, device=device)

            return loss, {"accuracy": accuracy}

        return evaluate