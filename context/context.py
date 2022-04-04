import torch
from sentence_transformers import SentenceTransformer
from util.logger import setup_logger
from util.nlp import dot_score, euclid_score
from dynaconf import settings
import math
import numpy as np


log = setup_logger(__name__)

torch.set_grad_enabled(False)


class ContextRetrieval:

    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.embedded_context = None
        self.contexts = None

    def embed(self, contexts):
        # Save context for later
        self.contexts = contexts
        # Chop up dem contexts into shards
        split_contexts, context_groupings = self._split_multiple_contexts(contexts, settings["max_sequence_length"])
        # Encode context shards
        sharded_contexts_emb = self.embedder.encode(split_contexts)
        # Average shards into context embeddings
        self.embedded_context = self._merge_split_context_embeddings(sharded_contexts_emb, context_groupings)

    # Some contexts have too many tokens to be parsed by the transformer, so we split them
    def _split_multiple_contexts(self, contexts, max_sequence_length):
        new_contexts = []
        # We keep track to which context the shards belong
        context_groupings = []
        for context in contexts:
            new_context = self._split_context(context, max_sequence_length)
            context_groupings.append((len(new_contexts), len(new_contexts)+(len(new_context)-1)))
            new_contexts += new_context
        return new_contexts, context_groupings

    # split context into equal sized shards shorter than max_sequence_length
    def _split_context(self, context, max_sequence_length):
        new_context = []
        num_shards = math.ceil(len(context)/max_sequence_length)
        shard_length = round(len(context)/num_shards)
        for i in range(num_shards):
            new_context.append(context[i*shard_length:(i+1)*shard_length])
        return new_context

    def _merge_split_context_embeddings(self, sharded_contexts_emb, context_groupings):
        contexts_emb = []
        for (idx_start, idx_end) in context_groupings:
            summed_emb = sum(sharded_contexts_emb[idx_start:idx_end + 1]) / ((idx_end - idx_start) + 1)
            contexts_emb.append(summed_emb)
        return np.array(contexts_emb)

    def retrieve_context(self, question, n=5):
        # Check if embedded context exists
        if self.embedded_context is None:
            log.error("ERROR: embed contexts before calling this function with .embed(context)")

        # Encode question
        query_emb = self.embedder.encode(question)

        # Compute score between query and all contexts embeddings
        # scores_ = dot_score(query_emb, self.embedded_context)[0].tolist()
        scores = euclid_score(query_emb, self.embedded_context).tolist()

        # Pair with contexts and sort by decreasing score
        contexts_score_pairs = sorted(list(zip(self.contexts, scores)), key=lambda x: x[1], reverse=False)

        # Output passages & scores
        return contexts_score_pairs[0:n]
