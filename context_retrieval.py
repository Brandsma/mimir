import torch.nn.functional as F
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np 
from util.logger import setup_logger

log = setup_logger(__name__)

torch.set_grad_enabled(False)


def chop_up_dem_contexts(contexts, max_sequence_length):
    new_contexts = []
    context_groupings = []
    for idx, context in enumerate(contexts):
        if len(context) <= max_sequence_length:
            context_groupings.append((len(new_contexts), len(new_contexts)))
            new_contexts.append(context)
            continue
        new_context = chop_up_the_context(context, max_sequence_length)
        context_groupings.append((len(new_contexts), len(new_contexts)+(len(new_context)-1))) # TODO This idx+count makes no sense (what if two contexts)
        new_contexts += new_context
    return new_contexts, context_groupings

def chop_up_the_context(context,  max_sequence_length):
    new_context = []
    if len(context) <= max_sequence_length:
        log.warn("Reached unreachable code in 'chop up the context'")
        return context
    else:
        i = max_sequence_length
        # Loop over all parts, and the remainder
        while i < len(context) + max_sequence_length:
            new_context.append(context[i - max_sequence_length:i])
            max_idx = i + max_sequence_length if i + max_sequence_length < len(context) else len(context)
            new_context.append(context[i:max_idx])
            i += max_sequence_length

        # NOTE: We chop up contexts all to a limit of 512. This means at the end of the context we can have a remainder
        # This remainder can be any size from 1 to 512. Later when we average the embeddings, both embeddings are 
        # weighted the same. This means that the smaller part is weighted advantageous. 
        # E.G. we have a context of length 513, we have one part of length 512 and one part of length 1.
        # When making the embeddings, combining the semantic vectors, the length of 1 is 50% of the averaged semantic vector.

        return new_context


def retrieve_context(question, contexts, n=5):
    #Load the model
    # model = SentenceTransformer('allenai/longformer-base-4096')
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    max_sequence_length = 512

    # Chop up dem contexts
    chopped_up_contexts, context_groupings = chop_up_dem_contexts(contexts, max_sequence_length)

    #Encode query and contexts using SentenceTransformer model.encode
    query_emb = model.encode(question)
    contexts_emb = model.encode(chopped_up_contexts)

    # Make larger sequence lengths objectively better
    better_contexts_emb = []
    for (idx_start, idx_end) in context_groupings:
        if idx_start == idx_end:
            better_contexts_emb.append(contexts_emb[idx_start])
            continue
            
        summed_emb = sum(contexts_emb[idx_start:idx_end + 1]) / ((idx_end - idx_start) + 1)

        better_contexts_emb.append(summed_emb)

    #Compute dot score between query and all contexts embeddings
    scores = dot_score(query_emb, better_contexts_emb)[0].tolist()

    #Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    #Sort by decreasing score
    contexts_score_pairs = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    return contexts_score_pairs[0:n]


if __name__=="__main__":
    retrieve_context()