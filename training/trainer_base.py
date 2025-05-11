import logging
import os
from typing import Dict, OrderedDict

from transformers import Trainer

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)


class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{test_key}":0
        })
        
    def log_best_metrics(self):
        pass
    
    
    def _maybe_log_save_evaluate():
        pass
    
    