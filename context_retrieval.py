import torch.nn.functional as F
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np 
from util.logger import setup_logger

log = setup_logger(__name__)

torch.set_grad_enabled(False)








if __name__=="__main__":
    ##retrieve_context()
    pass