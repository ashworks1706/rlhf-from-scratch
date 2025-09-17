import sys, os, logging
import transformers
from transformers import HfArgumentParser, TrainingArguments
from typing import List, Optional, Tuple
from dataclass import dataclass, field

class PPOArgs(TrainingArguments):

    actor_lr: Optional[float] = field(default = 1e-5)
    actor_weight_decay: Optional[float] = field(default = 0.0)

    ref_lr: Optional[float] = field(default = 1e-5)
    ref_weight_decay: Optional[float] = field(default = 0.0)

    reward_score_clip: Optional[float] = field(default = None)
    value_clip: Optional[float] = field(default = 0.2)
    ratio_clip: Optional[float] = field(default = 0.2)
