import numpy as np
# pip3 install pytorch-transformers
import torch
from pytorch_transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from pytorch_transformers.modeling_gpt2 import GPT2PreTrainedModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from torch import nn
from torch.nn import functional as F


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
        transformed_utterances = self.vectorizer.transform(utterances)
        informing_utterances = []
        for i, transformed_utterance in enumerate(transformed_utterances):
            similarities = linear_kernel(transformed_utterances[i:i + 1], self.knowledge_vectors).flatten()
            best_index = np.argmax(similarities)
            # the following lines would be for more than one returned utterance
            # best_index = np.argpartition(similarities, -self.num_output_sentences)[-self.num_output_sentences]
            # sorted_best_indices = sorted(best_indices.tolist(), key=lambda x: similarities[x], reverse=True)
            # informing_sents = [self.knowledge_utterances[j] for j in sorted_best_indices]
            informing_utterances.append(self.knowledge_utterances[best_index])
        return informing_utterances


class GPT2Generator(nn.Module):

    """
    Generates the next "word" (index) in a sequence.
    """

    def __init__(self):
        super(GPT2Generator, self).__init__()

        # TODO: can i make the outputs below medium and the knowledge medium?
        self.gpt2_config = GPT2Config.from_pretrained('gpt2-medium')
        self.input_gpt2 = GPT2Model.from_pretrained('gpt2-medium')
        self.gpt2_config_medium = GPT2Config.from_pretrained('gpt2-medium')
        self.output_gpt2 = GPT2Model.from_pretrained('gpt2-medium')
        self.lm_head = nn.Linear(self.gpt2_config.n_embd + self.gpt2_config.n_embd * 2,  # assume a single knowledge utterance for now
                                 self.gpt2_config_medium.vocab_size, bias=False)

        # initialize weights for ``self.lm_head``
        self.lm_head.weight.data.normal_(mean=0.0, std=self.gpt2_config.initializer_range)

        # tie weights
        wider_weights = torch.cat((self.input_gpt2.wte.weight.clone(),
                                   self.output_gpt2.wte.weight.clone(),
                                   self.output_gpt2.wte.weight.clone()), dim=1)
        self.lm_head.weight = nn.Parameter(wider_weights)

    def forward(self, scripture_sequences, knowledge_sequences, output_sequences):
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
        # make inferences on the scripture sequences with lh_model
        scripture_last_hidden_state, past = self.input_gpt2(scripture_sequences)  # see the GPT2LMHeadModel docstring
        scripture_last_hidden_layer = scripture_last_hidden_state[:, -1, :]

        # make inferences on the knowledge sequences with output_gpt2
        knowledge_last_hidden_state, past = self.output_gpt2(knowledge_sequences)  # see the GPT2LMHeadModel docstring
        knowledge_last_hidden_layer = knowledge_last_hidden_state[:, -1, :]

        # make inferences on output (predicted) sequences with output_gpt2
        output_last_hidden_state, past = self.output_gpt2(output_sequences)  # see the GPT2LMHeadModel docstring
        output_last_hidden_layer = output_last_hidden_state[:, -1, :]

        # concatenate the two outputs and connect to the final linear layer, which predicts which vocab index is next.
        concatenated = torch.cat((scripture_last_hidden_layer, knowledge_last_hidden_layer,
                                  output_last_hidden_layer), dim=1)
        tag_scores = F.softmax(self.lm_head(concatenated))  # take tag scores from the last layer
        return tag_scores
