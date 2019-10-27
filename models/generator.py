# pip3 install pytorch-transformers
from pytorch_transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from torch import nn


class GPT2Net(nn.Module):

    def __init__(self):
        super(GPT2Net, self).__init__()

        self.gpt2_config = GPT2Config.from_pretrained('gpt2-large')
        self.gpt2 = GPT2Model.from_pretrained('gpt2-large')
        self.gptlmh2 = GPT2LMHeadModel.from_pretrained('gpt2-large')

    def forward(self, sequences):
        last_hidden_state, past = self.gpt2(sequences)  # see the Model docstring
        tag_scores = last_hidden_state[:, -1, :]  # take tag scores from the last layer
        return tag_scores
