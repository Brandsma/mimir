from re import split
from collections import Counter
from torch import Tensor
import torch

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