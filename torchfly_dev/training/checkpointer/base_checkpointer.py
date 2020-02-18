import torch
from abc import ABC, abstractmethod

from typing import Union, Dict, Any, List, Tuple


class BaseCheckpointer(ABC):
    """
    An abstract class for checkpointer
    Checkpoint should be stored in the form of a dictionary with model
    and training states.
    """
    def __init__(self, models):
        self.models = models

    @abstractmethod
    def save_checkpoint(self, model_path):
        pass

    @abstractmethod
    def restore_checkpoint(self, model_path):
        pass

    @abstractmethod
    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        """
        pass