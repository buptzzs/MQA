import re
import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Dict

import gensim
import numpy as np
import torch
from deprecated import deprecated

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PRETRAINED_MODEL_ARCHIVE_MAP

import flair
from flair.nn import LockedDropout, WordDropout
from flair.data import Dictionary, Token, Sentence
from flair.file_utils import cached_path
from flair.embeddings import TokenEmbeddings

class BertEmbeddings(TokenEmbeddings):

    def __init__(self,
                 bert_model: str = 'bert-base-uncased',
                 bert_token: str = 'bert-base-uncased',
                 layers: str = '-1,-2,-3,-4',
                 pooling_operation: str = 'first'):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model: name of BERT model ('')
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()


        self.tokenizer = BertTokenizer.from_pretrained(bert_token)
        self.model = BertModel.from_pretrained(bert_model)
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.name = str(bert_model)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, token_subtoken_count):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(self, sentences, max_sequence_length: int) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0:(max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(BertEmbeddings.BertInputFeatures(
                unique_id=sentence_index,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                token_subtoken_count=token_subtoken_count))

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max([self.tokenizer.tokenize(sentence.to_tokenized_string()) for sentence in sentences], key=len))

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(sentences, longest_sentence_in_batch)
        all_input_ids = torch.LongTensor([f.input_ids for f in features])
        all_input_ids = all_input_ids.to(flair.device)
        all_input_masks = torch.LongTensor([f.input_mask for f in features])
        all_input_masks = all_input_masks.to(flair.device)

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        all_encoder_layers, _ = self.model(all_input_ids, token_type_ids=None, attention_mask=all_input_masks)

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu()[sentence_index]
                        all_layers.append(layer_output[token_index])

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == 'first':
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[token_idx:token_idx + feature.token_subtoken_count[token.idx]]
                        embeddings = [embedding.unsqueeze(0) for embedding in embeddings]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return len(self.layer_indexes) * self.model.config.hidden_size