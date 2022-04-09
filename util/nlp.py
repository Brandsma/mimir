from re import split
from collections import Counter
from torch import Tensor
import torch
from rouge import Rouge

torch.set_grad_enabled(False)

def split_sentences(text):
    parts = split('([.])', text)
    sentences = []

    for part in parts:
        if part == '':
            # remove the empty character after the last split
            continue

        if part == '.' and len(sentences) != 0:
            # add the period to the last added sentence
            sentences[len(sentences) - 1] = sentences[len(sentences) - 1] + '.'

        else :
            sentences.append(part)
    return sentences

def determine_most_frequent(list):
    occurence_count = Counter(list)
    most_common = occurence_count.most_common(1)[0][0]
    return most_common, occurence_count[most_common]


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
    # Compute the dot-product
    return torch.mm(a, b.transpose(0, 1))


def euclid_score(a: Tensor, b: Tensor):
    """
    Computes the euclidean distance between a and all vectors in b.
    Should be the same as FAISS
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    # Compute the euclidean distance
    m = torch.stack([a for _ in range(b.size()[0])])
    return torch.sqrt(torch.sum(torch.square(torch.sub(m, b)), 1))

def edit_distance(a: str, b: str):
    """
    Computes the edit distance between two strings.
    The edit distance is the amount of edits required
    to transform string a to string b.
    """
    if len(a) > len(b):
        difference = len(a) - len(b)
        a[:difference]
    elif len(b) > len(a):
        difference = len(b) - len(a)
        b[:difference]
    else:
        difference = 0

    for i in range(len(a)):
        if a[i] != b[i]:
            difference += 1

    return difference

def rouge_l(a: str, b:str):
    """
    Calculates the ROUGE_L score. 
    The ROUGE_L score measures the longest common subsequence between two strings.
    Scores is a dictionary with the F1 score, precision and recall 
    for keys 'f','p' and 'r' respectively.
    Only the F1 score is returned. 
    """
    rouge = Rouge()
    scores = rouge.get_scores(a,b)
    return scores['rouge-l']['f']