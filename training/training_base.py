import logging
import os
from typing import Dict, OrderedDict

from transformers import Trainer

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)