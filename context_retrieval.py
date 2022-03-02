import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import torch
torch.set_grad_enabled(False)

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    Taken from the SentenceTransformer library
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    print(a.shape, b.shape)
    # Compute the dot-product
    return torch.mm(a, b.transpose(0, 1))

def retrieve_context(question, contexts, n=5):
    #Load the model
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    #Encode query and contexts using SentenceTransformer model.encode
    query_emb = model.encode(question)
    contexts_emb = model.encode(contexts)

    #Compute dot score between query and all contexts embeddings
    scores = dot_score(query_emb, contexts_emb)[0].tolist()

    #Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    #Sort by decreasing score
    contexts_score_pairs = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    return contexts_score_pairs[0:n]

def retrieve_context_FAISS(question, contexts, n=5):
    # Load the model
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # Add FAISS indexing
    contexts_emb = contexts.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example['text'], return_tensors="pt"))[0][0].numpy()})
    contexts_emb.add_faiss_index(column='embeddings')

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    query_emb = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
    scores, contexts = contexts_emb.get_nearest_examples('embeddings', query_emb, k=10)

    #Compute dot score between query and all contexts embeddings
    scores = dot_score(query_emb, contexts_emb)[0].tolist()

    #Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    #Sort by decreasing score
    contexts_score_pairs = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    return contexts_score_pairs[0:n]

if __name__=="__main__":
    retrieve_context()