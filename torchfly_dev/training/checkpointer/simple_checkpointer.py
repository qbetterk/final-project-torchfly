import os
import logging
import torch
import time
from .base_checkpointer import BaseCheckpointer

from typing import Union, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class SimpleCheckpointer(BaseCheckpointer):
    """
    Simple Checkpointer implements the basic functions
    """
    def __init__(self,
                 model,
                 num_checkpoints_to_keep: int = 1000,
                 keep_checkpoint_every_num_seconds: float = 3600,
                 storage_dir: str = "Checkpoints"):
        self.model = model
        self.storage_dir = storage_dir
        self.current_checkpoint = {}
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.keep_checkpoint_every_num_seconds = keep_checkpoint_every_num_seconds
        self._saved_checkpoint_paths = []
        self._last_checkpoint_time = time.time()
        # initialization
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_checkpoint(self, epoch, model_state, training_state):
        checkpoint_path = os.path.join(self.storage_dir,
                                       f"model_state_epoch_{epoch}.pth")
        self.current_checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "training_state": training_state
        }

        if self.num_checkpoints_to_keep > 0:
            self._saved_checkpoint_paths.append((time.time(), checkpoint_path))
            path_to_remove = self._saved_checkpoint_paths.pop(0)

            # check time requirement
            remove_path = True
            if self.keep_checkpoint_every_num_seconds is not None:
                save_time = path_to_remove[0]
                time_since_checkpoint_kept = (save_time -
                                              self._last_checkpoint_time)
                if time_since_checkpoint_kept > self.keep_checkpoint_every_num_seconds:
                    # We want to keep this checkpoint.
                    remove_path = False
                    self._last_checkpoint_time = save_time

            if remove_path:
                for fname in path_to_remove[1:]:
                    if os.path.isfile(fname):
                        os.remove(fname)

        torch.save(self.current_checkpoint, checkpoint_path)

    def restore_checkpoint(self):
        latest_checkpoint_path = self.find_latest_checkpoint()
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
        return checkpoint

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest checkpoint file.
        """
        return {}
