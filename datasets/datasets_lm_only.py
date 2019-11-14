import os
import re

import torch
from pytorch_transformers import GPT2Tokenizer
from torch.utils.data import Dataset


class CustomTokenizer(GPT2Tokenizer):

    """
    Quiet the protestations of GPT2Tokenizer when the number of tokens exceeds max max_dataset_length,
    and optimize a bit by stopping tokenization when number of tokens exceeds max max_dataset_length.
    """

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
        (resp. a sequence of ids), using the vocabulary.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for i, token in enumerate(tokens):
            ids.append(self._convert_token_to_id_with_added_voc(token))

        return ids


class BibleCommentaryDataset(Dataset):

    tokenizer = CustomTokenizer.from_pretrained('gpt2-large')
    eos_index = tokenizer.encode(tokenizer.eos_token)[0]

    def __init__(self, dir_='trainingdata', filenames=['Beal.txt'], min_sequence_length=10,
                 max_sequence_length=300):
        text = ''
        for filename in filenames:
            with open(os.path.join(dir_, filename)) as f:
                text += (' ' + f.read())

        self.text = re.sub(r'\s{2,}', ' ', text)
        self.sequence = self.tokenizer.encode(self.text)

        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.current_sequence_length = min_sequence_length

    def __getitem__(self, item):
        x = torch.Tensor(self.sequence[item:item + self.current_sequence_length]).long()
        y = self.sequence[item + self.current_sequence_length]
        return x, y

    def __len__(self):
        return len(self.sequence) - self.current_sequence_length
