import numpy as np
# pip3 install pytorch-transformers
import torch
from pytorch_transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from pytorch_transformers.modeling_gpt2 import GPT2PreTrainedModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from torch import nn


class CPULinear:

    """
    A model that returns n most important knowledge sentences from a corpus
    for a given batch of utterances.

    NOTE WELL: only runs on CPU and is based upon scikitlearn, not pytorch.
    """

    def __init__(self, num_output_sentences=1, knowledge_utterances=[]):
        """
        Constructor.

        Parameters
        ----------
        num_output_sentences (int) - number of sentences to output from knowledge base.
        knowledge_utterances (list of str) - utterances that exhibit subject area knowledge.
        """
        self.num_output_sentences = num_output_sentences
        self.vectorizer = TfidfVectorizer()
        self.knowledge_utterances = knowledge_utterances
        self.knowledge_vectors = self.vectorizer.fit_transform(knowledge_utterances)

    def forward(self, utterances):
        """
        Infers ``self.output_sentences`` most important knowledge utterances to
        inform input utterances.

        Parameters
        ----------
        utterances (list of str) - batch of utterances to gather most important sentences for.

        Returns
        -------
        (list of str--2 dims.) list of list of most important knowledge utterances.
        """
        transformed_utterances = self.vectorizer.transform(utterances.tolist())
        informing_utterances = []
        for i, transformed_utterance in enumerate(transformed_utterances):
            similarities = linear_kernel(transformed_utterances[i:i + 1], self.knowledge_vectors).flatten()
            best_indices = np.argpartition(similarities, -self.num_output_sentences)[-self.num_output_sentences]
            sorted_best_indices = sorted(best_indices, key=lambda x: similarities[x], reverse=True)
            informing_sents = [self.knowledge_utterances[j] for j in sorted_best_indices]
            informing_utterances.append(informing_sents)
        return informing_utterances


class GPT2Generator(nn.Module):

    """
    Generates the next "word" (index) in a sequence.
    """

    def __init__(self):
        super(GPT2Generator, self).__init__()

        self.gpt2_config = GPT2Config.from_pretrained('gpt2-large')
        self.conversation_gpt2 = GPT2Model.from_pretrained('gpt2-large')
        self.knowledge_gpt2 = GPT2Model.from_pretrained('gpt2-large')
        self.lm_head = nn.Linear(self.gpt2_config.n_embd * 2,  # assume a single knowledge utterance for now
                                 self.gpt2_config.vocab_size, bias=False)

    def forward(self, conversation_sequences, knowledge_sequences):
        """
        Makes an inference.

        Parameters
        ----------
        sequences (torch.Tensor) - batch of samples.
        knowledge_seqs (torch.Tensor) - batch of corresponding knowledge sequences
                                        as determined by a CPULinear model.

        Returns
        -------
        Tag scores (torch.Tensor) that indicate most likely next word for sample in batch.
        """
        # make inferences on the conversation sequences
        conversation_last_hidden_state, past = self.conversation_gpt2(conversation_sequences)  # see the GPT2LMHeadModel docstring
        conversation_last_hidden_layer = conversation_last_hidden_state[:, -1, :]

        # make inferences on the knowledge sequences
        knowledge_last_hidden_state, past = self.knowledge_gpt2(knowledge_sequences)  # see the GPT2LMHeadModel docstring
        knowledge_last_hidden_layer = knowledge_last_hidden_state[:, -1, :]

        # concatenate the two outputs and connect to the final linear layer, which predicts which vocab index is next.
        concatenated = torch.cat(knowledge_last_hidden_layer, conversation_last_hidden_layer)
        tag_scores = self.lm_head(concatenated)  # take tag scores from the last layer
        return tag_scores
