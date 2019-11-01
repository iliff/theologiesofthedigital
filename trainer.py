# TODO: change for this bible project!

import os

import torch  # pip3 install torch
# pip3 install pytorch-transformers
from pytorch_transformers import AdamW
from torch import nn
from torch.utils.data import DataLoader

from datasets.datasets import BibleCommentaryDataset
from models.generator import CPULinear, GPT2Generator


def train(model_filename='verse_continuation_model.pt',
          lr=6.5e-5, correct_bias=False, epochs=1000, inferencehook=None,
          sample_sentences=[]):

    dataset = BibleCommentaryDataset(max_seq_len=512, dataset_length=1_000, max_df_len=100)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                            num_workers=1)

    tfidf_model = CPULinear(num_output_sentences=1, knowledge_utterances=dataset.df.comment)

    if model_filename and os.path.exists(model_filename):
        print('loading model ...')
        model = torch.load(model_filename)
    else:
        model = GPT2Generator()
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    last_loss = None

    for epoch_i, epoch in enumerate(range(epochs)):

        running_losses = []

        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            X_gpt2, X_tfidf, y = batch

            # convert X_tfidf to sequences
            X_tfidf = tfidf_model.forward(X_tfidf)
            sequenced_X_tfidf = torch.Tensor([(dataset.tokenizer.encode(x) + [0] * 200)[:200] for x in X_tfidf]).long()

            # push X and y to cuda
            X_gpt2 = X_gpt2.to('cuda')
            X_tfidf = X_tfidf.to('cuda')
            y = y.to('cuda')

            predictions = model(X_gpt2, X_tfidf)
            loss = criterion(predictions, y)
            running_losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()

            if i % 50 == 0:
                print('EPOCH {}, current_sentence_length {}, Batch {}: loss == {:.8f}'
                      .format(epoch, dataset.sentence_length, i,
                              sum(running_losses) / len(running_losses)))
                running_losses = []

            if i % 1000 == 0:
                if inferencehook:
                    inferencehook(dataset, model, sentences=sample_sentences)
                if last_loss is None:
                    last_loss = loss
                elif last_loss > loss:
                    print('Saving {} with loss {:.8f}'.format(model_filename, loss))
                    torch.save(model, 'modeldata/' + model_filename)
                    last_loss = loss


if __name__ == '__main__':
    train(model_filename='verse_continuation_model.pt',
          lr=6.5e-5, correct_bias=False, epochs=1000, inferencehook=None,
          sample_sentences=[])
