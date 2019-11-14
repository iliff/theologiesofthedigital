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

    def __init__(self, output_sent_indices_to_join=[1, 3], knowledge_utterances=[]):
        """
        Constructor.

        Parameters
        ----------
        output_sent_indices_to_join (int) - number of sentences to output from knowledge base.
        knowledge_utterances (list of str) - utterances that exhibit subject area knowledge.
        """
        self.output_sent_indices_to_join = output_sent_indices_to_join
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

            # add together sentences identified in self.output_sent_indices_to_join
            for j in range(max(self.output_sent_indices_to_join) + 1):
                best_index = np.argmax(similarities)
                if j in self.output_sent_indices_to_join:
                    informing_utterances.append(self.knowledge_utterances[best_index])
                similarities[best_index] = 0.

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
