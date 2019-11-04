# pip3 install pytorch-transformers
from pytorch_transformers import GPT2LMHeadModel, GPT2Config
from torch import nn


class GPT2Generator(nn.Module):

    """
    Generates the next "word" (index) in a sequence.
    """

    def __init__(self):
        super(GPT2Generator, self).__init__()

        # TODO: can i make the outputs below large and the knowledge medium?
        self.gpt2_config = GPT2Config.from_pretrained('gpt2-large')
        self.lh_model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    def forward(self, scripture_sequences):
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
        last_hidden_state, past = self.lh_model(scripture_sequences)
        tag_scores = last_hidden_state[:, -1, :]

        return tag_scores
