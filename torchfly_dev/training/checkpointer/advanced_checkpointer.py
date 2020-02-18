import os
import glob
import logging
import torch
import time
from .base_checkpointer import BaseCheckpointer

from typing import Union, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class AdavancedCheckpointer(BaseCheckpointer):
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

    def save_checkpoint(self, stamp:str, state: Dict[str, Any]) -> None:
        """
        Args:
            stamp: A string to identify the checkpoint. It can just be the epoch number
            state: A dictionary to store all necessary information for later restoring

        """
        checkpoint_path = os.path.join(self.storage_dir,
                                       f"{stamp}_state.pth")

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

    def restore_checkpoint(self, search_method=None):
        """
        Args:
            search_method: a Callable to find the wanted checkpoint path
        """
        # if not specified
        if not search_method:
            search_method = self.find_latest_checkpoint

        checkpoint_path = search_method()
        # map to the cpu first instead of error
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        return checkpoint

    def find_latest_checkpoint(self) -> str:
        """
        Return the path of the latest checkpoint file.
        """
        files = glob.glob(os.path.join(self.storage_dir, "*_state.pth"))
        latest_file_path = max(files, key=os.path.getctime)
        latest_file_path = os.path.join(self.storage_dir, latest_file_path)
        return latest_file_path
