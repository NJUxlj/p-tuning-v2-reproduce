import logging
import os
import random
import sys

from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union
import math
import random
import time
import warnings
import collections


from transformers.debug_utils import DebugOption, DebugUnderflowOverflow



from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_torch_tpu_available,   # 过时了
    is_torch_available,
)
