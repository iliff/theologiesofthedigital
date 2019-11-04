import os
import multiprocessing

import numpy as np
import pandas as pd
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
            if i >= self.max_len:
                break
            ids.append(self._convert_token_to_id_with_added_voc(token))

        return ids


class BibleCommentaryDataset(Dataset):

    tokenizer = CustomTokenizer.from_pretrained('gpt2-large')
    eos_index = tokenizer.encode(tokenizer.eos_token)[0]

    def __init__(self, dir_='trainingdata', max_seq_len=512, archive_filename='commentaries',
                 refresh=False, max_dataset_length=10_000, batches_per_sent_len=20,
                 limit_df_to=None, df_book=None):
        self.dir_ = dir_
        self.max_seq_len = max_seq_len
        self.archive_filename = archive_filename
        self.max_dataset_length = max_dataset_length
        self.batches_per_sent_len = batches_per_sent_len

        self.current_sample = pd.DataFrame()
        self.has_called_length = False
        self.len_calls = 0

        if not refresh and archive_filename in os.listdir('trainingdataarchived'):
            self.df = pd.read_csv(os.path.join('trainingdataarchived', archive_filename))
        else:
            self.df = self._construct_df(archive_filename, dir_, max_df_len=limit_df_to,
                                         df_book=df_book)
            self.df.to_csv(os.path.join('trainingdataarchived', archive_filename) + '.csv',
                           index=False)

        self.sentence_length = 0

        # get first sample and first sentence length
        while not len(self.current_sample):
            self.set_current_sample()
            self.sentence_length = self.sentence_length + 1 if self.sentence_length < 1024 else 0

    def _add_sequence_lengths_to_df(self, df):
        df['verse_token_length'] = df['verse_sequence'].apply(lambda x: len(x))
        df['comment_token_length'] = df['comment_sequence'].apply(lambda x: len(x))
        df['total_token_length'] = df['verse_token_length'] + df['comment_token_length']
        return df

    def _add_sequences_to_df(self, df):
        df['verse_sequence'] = df['verse'].apply(lambda x: self.tokenizer.encode(x))
        df['comment_sequence'] = df['comment'].apply(lambda x: self.tokenizer.encode(x[:512]))
        return df

    def _clean_df(self, df):
        df = df.dropna(subset=['comment'])
        df.loc[:, 'comment'] = df['comment'].apply(lambda x: x.strip())
        df = df[df['comment'] != '']
        return df

    def _construct_df(self, archive_filename, dir_, max_df_len=None,
                      df_book=None):
        print('constructing initial df ...')
        df = pd.DataFrame(columns=['reference', 'verse', 'comment'])
        if df_book:
            df = df[df['reference'].str.contains(df_book)]
        for fn in os.listdir(dir_):
            subdf = pd.read_csv(os.path.join(dir_, fn), sep='|', header=None,
                                error_bad_lines=False, names=['reference', 'verse', 'comment'])

            df = df.append(subdf, ignore_index=True, sort=False)
        print('cleaning df ...')
        df = self._clean_df(df)
        if max_df_len:
            df = df.sample(frac=1.).iloc[:max_df_len]
        # df.to_csv(os.path.join('../trainingdataarchived', self.archive_filename + '_raw.csv'))
        print('adding sequences to df')
        df = self._add_sequences_to_df(df)
        df = self._add_sequence_lengths_to_df(df)
        df = df.sort_values(by=['total_token_length'], ascending=True)
        return df

    def __getitem__(self, item):
        verse_sequence = self.current_sample.iloc[item]['verse_sequence']
        comment_sequence = self.current_sample.iloc[item]['comment_sequence']
        nn_x = torch.Tensor(verse_sequence + comment_sequence)[:self.sentence_length].long()
        nn_y = (verse_sequence + comment_sequence)[self.sentence_length]
        return (nn_x, nn_y)

    def __len__(self):
        return min(len(self.current_sample), self.max_dataset_length) or 1

    def set_current_sample(self):
        df = self.df[(self.df['comment_token_length'] > self.sentence_length)]
        self.current_sample = df.sample(n=min(self.max_dataset_length, len(df)), replace=False).reset_index()

    def set_sentence_length(self, value):
        self.sentence_length = value


if __name__ == '__main__':
    commentary_dataset = BibleCommentaryDataset()
    print(len(commentary_dataset))
    for j in iter(range(len(commentary_dataset))):
        print(j, commentary_dataset[j])
    print(len(commentary_dataset))
    # the following should be longer
    for j in iter(range(len(commentary_dataset))):
        print(j, commentary_dataset[j])
