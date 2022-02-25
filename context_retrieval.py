import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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


#Mean Pooling - Average all the embeddings produced by the model
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    # Expand the mask to the same size as the token embeddings to avoid indexing errors
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Compute the mean of the token embeddings
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(model, tokenizer, texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, return_tensors='pt') # Your code here
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings


def retrieve_context(question, contexts, n=1):
    # Load the model and tokenizer from HuggingFace Hub
    model_name = "google/bigbird-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    #Encode query and contexts with the encode function
    query_emb = encode(model, tokenizer, question)
    contexts_emb = encode(model, tokenizer, contexts)

    #Compute dot score between query and all contexts embeddings
    scores = torch.mm(query_emb, contexts_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    #Sort by decreasing score
    contexts_score_pairs = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)

    #Return n contexts
    return contexts_score_pairs[0:n][0]

if __name__=="__main__":
    retrieve_context()