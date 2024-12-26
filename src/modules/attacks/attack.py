from abc import ABC, abstractmethod

import torch

class Attack(ABC):
    """
    A generic interface for attack
    """

    @abstractmethod
    def on_batch_selection(self, inputs: torch.Tensor, labels: torch.Tensor):
        
        return inputs, labels

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss

