import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np 
from util.logger import setup_logger
from nlp_util import dot_score
from dynaconf import settings


log = setup_logger(__name__)

torch.set_grad_enabled(False)

class Context:

    def __init__(self):
        pass

    def split_multiple_contexts(self, contexts, max_sequence_length):
        # Some contexts have too many tokens to be parsed by the transformer, so we split them
        new_contexts = []
        # We keep track to which context the shards belong
        context_groupings = []
        for context in contexts:
            if len(context) <= max_sequence_length:
                context_groupings.append((len(new_contexts), len(new_contexts), 0))
                new_contexts.append(context)
                continue
            new_context, remainder = self.split_context(context, max_sequence_length)
            context_groupings.append((len(new_contexts), len(new_contexts)+(len(new_context)-1), remainder))
            new_contexts += new_context
        return new_contexts, context_groupings

    def split_context(self, context, max_sequence_length):
        new_context = []
        if len(context) <= max_sequence_length:
            log.warn("Reached unreachable code in 'split_context'")
            return context
        else:
            i = 0
            # Loop over all parts, and the remainder
            while i < len(context):
                max_idx = i + max_sequence_length if i + max_sequence_length < len(context) else len(context)
                new_context.append(context[i:max_idx])
                i += max_sequence_length
            
            # NOTE: We chop up contexts all to a limit of 512. This means at the end of the context we can have a remainder
            # This remainder can be any size from 1 to 512. Later when we average the embeddings, both embeddings are 
            # weighted the same. This means that the smaller part is weighted advantageous. 
            # E.G. we have a context of length 513, we have one part of length 512 and one part of length 1.
            # When making the embeddings, combining the semantic vectors, the length of 1 is 50% of the averaged semantic vector.
            # remainder is the length of the last split, all other splits have the max_sequence_length
            remainder = len(new_context[len(new_context) - 1])
        return new_context, remainder

    def merge_split_context_embeddings(self, sharded_contexts_emb, context_groupings):
        contexts_emb = []
        # TODO: Not all combinations have a remainder!! If they're not split, they never get a remainder.
        for (idx_start, idx_end, remainder) in context_groupings:
            # If the group is of size 1, it was not split and we can append it in its place.
            if idx_start == idx_end:
                contexts_emb.append(sharded_contexts_emb[idx_start])
                continue
            # If the group is larger, average the semantic vectors and append in its place
            weighted_average_embeddings = self.embeddings_weighted_average(sharded_contexts_emb[idx_start:idx_end + 1], remainder)
            #summed_emb = sum(sharded_contexts_emb[idx_start:idx_end + 1]) / ((idx_end - idx_start) + 1)
            contexts_emb.append(weighted_average_embeddings)
        return contexts_emb

    def embeddings_weighted_average(self, sharded_context_emb, remainder):
        # TODO implement somewhere a little above. Need to get the remainder elegantly
        shards = len(sharded_context_emb)
        # If there is one shard, we don't need to average
        if shards == 1:
            return sharded_context_emb

        max_sequence_length = settings["max_sequence_length"]

        # Scale all shards except for the final one to the max_sequence_length
        summed_emb = sum(sharded_context_emb[:shards - 2]) * max_sequence_length
        # Scale the final shard with its remainder and add it to the sums
        summed_emb += sharded_context_emb[shards - 1] * remainder
        # Get the average by dividing the summed_embedding by the 
        averaged_emb = summed_emb / (((shards - 1) * max_sequence_length) + remainder)
        return averaged_emb

    def retrieve_context(self, question, contexts, n=5):
        #Load the model
        # model = SentenceTransformer('allenai/longformer-base-4096')
        model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        max_sequence_length = settings["max_sequence_length"]

        # Chop up dem contexts
        split_contexts, context_groupings = self.split_multiple_contexts(contexts, max_sequence_length)

        #Encode query and contexts using SentenceTransformer model.encode
        query_emb = model.encode(question)
        
        # First encode all context shards
        sharded_contexts_emb = model.encode(split_contexts)

        # Then collaps the semantic vectors / embeddings of sharded context groups 
        #   to retrieve a single embedding for a context that was too long.
        contexts_emb = self.merge_split_context_embeddings(sharded_contexts_emb, context_groupings)

        #Compute dot score between query and all contexts embeddings
        scores = dot_score(query_emb, contexts_emb)[0].tolist()

        #Combine contexts & scores
        contexts_score_pairs = list(zip(contexts, scores))

        #Sort by decreasing score
        contexts_score_pairs = sorted(contexts_score_pairs, key=lambda x: x[1], reverse=True)

        #Output passages & scores
        return contexts_score_pairs[0:n]