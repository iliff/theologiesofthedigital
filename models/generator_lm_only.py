import re

from gensim.summarization import keywords
import numpy as np
# pip3 install pytorch-transformers
from pytorch_transformers import GPT2LMHeadModel, GPT2Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from torch import nn


class CPULinear:
    """
    A model that returns n most important knowledge sentences from a corpus
    for a given batch of utterances.

    NOTE WELL: only runs on CPU and is based upon scikitlearn, not pytorch.
    """

    def __init__(self, number_of_sentences_as_seed=2, knowledge_utterances=[]):
        """
        Constructor.

        Parameters
        ----------
        number_of_sentences_as_seed (int) - number of sentences to output from knowledge base.
        knowledge_utterances (list of str) - utterances that exhibit subject area knowledge.
        """
        self.number_of_sentences_as_seed = number_of_sentences_as_seed
        self.vectorizer = TfidfVectorizer()
        self.knowledge_utterances = knowledge_utterances
        self.knowledge_vectors = self.vectorizer.fit_transform(knowledge_utterances)

        # compile passage keyterms for help honing in on a passage
        with open('../trainingdata/nrsv.txt') as f:
            rev_text = f.read()
        rev_text_lines = rev_text.split('\n')
        self.passage_kw_clusters = {}
        for i in range(0, len(rev_text_lines), 7):
            subtext_lines = rev_text_lines[i:i + 7]
            subtext = ' '.join(subtext_lines)
            key_terms = keywords(subtext, ratio=0.2, split=True)
            if key_terms:
                self.passage_kw_clusters[subtext] = set(key_terms)

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

            # find cluster set closest to utterance
            best_passage, best_cluster_score = '', 0
            for passage, key_term_cluster in self.passage_kw_clusters.items():
                cluster_score = len(set(re.findall(r'\w+', utterances[i])) & key_term_cluster)
                if cluster_score > best_cluster_score:
                    key_terms = key_term_cluster
                    best_cluster_score = cluster_score

            similarities = linear_kernel(transformed_utterances[i:i + 1], self.knowledge_vectors).flatten()

            # add together sentences identified in self.number_of_sentences_as_seed
            attempts = 0
            first_best_index = np.argmax(similarities)  # as default in case no others are found
            while len(informing_utterances) < self.number_of_sentences_as_seed:
                best_index = np.argmax(similarities)
                utterance = self.knowledge_utterances[best_index]
                lower_utterance = utterance.lower()
                for key_term in key_terms:
                    if key_term in lower_utterance:
                        informing_utterances.append(utterance)
                        break
                similarities[best_index] = 0.
                if attempts > 25:
                    if len(informing_utterances) == 0:
                        informing_utterances.append(self.knowledge_utterances[first_best_index])
                    break
                attempts += 1

        return ' '.join(informing_utterances)


class GPT2Generator(nn.Module):
    """
    Generates the next "word" (index) in a sequence.
    """

    def __init__(self):
        super(GPT2Generator, self).__init__()

        # TODO: can i make the outputs below large and the knowledge medium?
        self.gpt2_config = GPT2Config.from_pretrained('gpt2-large')
        self.lh_model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    def forward(self, sequences):
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
        last_hidden_state, past = self.lh_model(sequences)
        tag_scores = last_hidden_state[:, -1, :]

        return tag_scores
