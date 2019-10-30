import os
import multiprocessing

import numpy as np
import pandas as pd
from pytorch_transformers import GPT2Tokenizer
from torch.utils.data import Dataset


class CustomTokenizer(GPT2Tokenizer):

    """
    Quiet the protestations of GPT2Tokenizer when the number of tokens exceeds max length,
    and optimize a bit by stopping tokenization when number of tokens exceeds max length.
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

    def __init__(self, dir_='../trainingdata', max_seq_len=512, archive_filename='commentaries',
                 refresh=False):
        self.dir_ = dir_
        self.max_seq_len = max_seq_len
        self.eos_index = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        if not refresh and archive_filename in os.listdir('../trainingdataarchived'):
            self.df = pd.read_csv(os.path.join('../trainingdataarchived', archive_filename))
        else:
            self.df = self._construct_df(archive_filename, dir_)
            self.df.to_csv(os.path.join('../trainingdataarchived', archive_filename),
                           index=False)

    def _clean_df(self, df):
        df = df.dropna(subset=['comment'])
        df.loc[:, 'comment'] = df['comment'].apply(lambda x: x.strip())
        df = df[df['comment'] != '']
        return df

    def _construct_df(self, archive_filename, dir_):
        print('constructing initial df ...')
        df = pd.DataFrame(columns=['reference', 'verse', 'comment'])
        for fn in os.listdir(dir_):
            subdf = pd.read_csv(os.path.join(dir_, fn), sep='|', header=None,
                                error_bad_lines=False, names=['reference', 'verse', 'comment'])

            df = df.append(subdf, ignore_index=True, sort=False)
        print('cleaning df ...')
        df = self._clean_df(df)
        print('converting df to text tokens')
        df = self._convert_df_to_text_tokens(df)
        print('constructing making df token gradations ...')
        df = self._make_token_gradations(df)
        df['length'] = df['sequence'].apply(lambda x: len(x))
        return df.sort_values(by='length', ascending=True)

    def _convert_df_to_text_tokens(self, df):
        df['text'] = df['verse'] + ' {} '.format(self.tokenizer.eos_token) + df['comment']
        df['tokens'] = df['text'].apply(lambda x: self.tokenizer.encode(x)[:self.max_seq_len])
        df = df[['text', 'tokens']]
        return df

    def _make_token_gradations(self, df):
        print('splitting into multiple dfs for multiprocessing.')
        dfs = np.array_split(df, 14)
        manager = multiprocessing.Manager()
        processed_dfs = manager.list()
        for i, subdf in enumerate(dfs):
            print('starting subprocessing on df', i)
            p = multiprocessing.Process(target=_create_df_with_token_gradations, args=(subdf, processed_dfs))
            p.start()
            p.join()
        print('joining multiprocessed dfs together')
        new_df = pd.DataFrame()
        for subdf in processed_dfs:
            new_df = new_df.append(subdf, ignore_index=True)
        return new_df

    def __getitem__(self, item):
        return self.df[['sequence', 'next_token', 'text', 'sequence']].iloc[item].tolist()

    def __len__(self):
        return len(self.df)


def _create_df_with_token_gradations(df, processed_dfs):
    new_df = pd.DataFrame(columns=['text', 'next_word', 'sequence', 'next_token'])
    for i, row in df.iterrows():
        start_of_string_index = row['tokens'].index(BibleCommentaryDataset.tokenizer.eos_index) + 1
        for j in range(start_of_string_index, len(row['tokens']) - 1):
            new_df = new_df.append({
                'sequence': row['tokens'][:j],
                'next_token': row['tokens'][j],
                'text': BibleCommentaryDataset.tokenizer.decode(row['tokens'][:j]),
                'next_word': BibleCommentaryDataset.tokenizer.decode(row['tokens'][j]),
            }, ignore_index=True)
    processed_dfs.append(new_df)
    print('completed a subdf')


if __name__ == '__main__':
    commentary_dataset = BibleCommentaryDataset()
    print(commentary_dataset[3])
    print(commentary_dataset[4])
    print(len(commentary_dataset))
