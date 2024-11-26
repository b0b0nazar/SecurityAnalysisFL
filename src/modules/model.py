from torch import nn, optim
from torchvision import models
import torch

def mobilenet_v2(**kwargs):
    """Create a MobileNetV2 model with the specified number of classes."""
    num_classes = kwargs.get('num_classes', 8)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def resnet101(**kwargs):
    """Create a ResNet101 model with the specified number of classes."""
    num_classes = kwargs.get('num_classes', 8)
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model = torch.nn.Sequential(
        *(list(model.children())[:-1]), torch.nn.Flatten(), nn.Linear(in_features=2048, out_features=num_classes)
    )
    # Set the hidden_dimension
    model.hidden_dimension = 2048
    return model

class ModelFactory:
    @staticmethod
    def create_model(config):
        """
        Create a model based on the configuration provided.

        Args:
            config (dict): Configuration dictionary containing model details.

        Returns:
            torch.nn.Module: The created model instance.

        Raises:
            ValueError: If the model name is unknown.
        """

        model_name = config.model.name
        if model_name == 'mobilenet_v2':
            return mobilenet_v2(**config.model)
        if model_name == 'resnet101':
            return resnet101(**config.model)
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))