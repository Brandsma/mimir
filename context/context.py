import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np 
from util.logger import setup_logger
from nlp_util import dot_score

log = setup_logger(__name__)

torch.set_grad_enabled(False)

class Context:

    def __init__():
        pass

