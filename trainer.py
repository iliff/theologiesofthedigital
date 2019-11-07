import os
import random

import torch  # pip3 install torch
# pip3 install pytorch-transformers
from pytorch_transformers import AdamW
from torch import nn
from torch.utils.data import DataLoader

from datasets.datasets import BibleCommentaryDataset
from models.generator import CPULinear, GPT2Generator
from traininghooks import generatorhook


def train(model_filename='verse_continuation_model.pt',
          lr=6.5e-5, correct_bias=False, epochs=1000, inferencehook=None,
          num_sentences=4, optimize_every=1, load_model=None):
    dataset = BibleCommentaryDataset(max_seq_len=512, max_dataset_length=300,
                                     batches_per_sent_len=2, df_book=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=1)

    tfidf_model = CPULinear(num_output_sentences=1, knowledge_utterances=dataset.df.comment.tolist())

    if model_filename and os.path.exists(model_filename):
        print('loading model ...')
        model = torch.load(model_filename)
    elif load_model:
        print('loading model {} ...'.format(load_model))
        model = torch.load(os.path.join('modeldata', load_model))
    else:
        model = GPT2Generator()

    model = model.to('cuda')

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.AdaptiveLogSoftmaxWithLoss(model.gpt2_config.vocab_size,
                                              model.gpt2_config.vocab_size,
                                              [50, 200, 2000]).to('cuda')

    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    optimizer.zero_grad()

    epoch_losses = []
    last_saved_epoch_loss = None

    optimize_counter = 0

    for epoch_i, epoch in enumerate(range(epochs)):

        dataset.set_sentence_length(151)
        dataset.set_current_sample()

        running_losses = []

        for i, batch in enumerate(dataloader):

            X_scripture, X_comment, X_tfidf, y = batch

            # convert X_tfidf to sequences
            X_tfidf = tfidf_model.forward(X_comment)
            # the following will break if batch size > 1
            sequenced_X_tfidf = torch.Tensor([dataset.tokenizer.encode(x)[:50] for x in X_tfidf]).long()

            # push X and y to cuda
            X_scripture = X_scripture.to('cuda')
            X_comment = X_comment.to('cuda')
            X_tfidf = sequenced_X_tfidf.to('cuda')
            y = y.to('cuda')

            for j in range(1, 151):
                predictions = model(X_scripture, X_comment[:, :j], X_tfidf)
                # AdaptiveLogSoftmaxWithLoss returns a NamedTuple with output and loss attributes
                loss = criterion(predictions, X_comment[:, j] if j < 150 else y).loss / optimize_every
                running_losses.append(loss.item())
                epoch_losses.append(loss.item())
                loss.backward()

                if j % 10 == 0:
                    print('EPOCH {}, current_sentence_length {}, Batch {}: loss == {:.8f}'
                          .format(epoch, j, i,
                                  sum(running_losses) * optimize_every / len(running_losses)))
                    running_losses = []

                nn.utils.clip_grad_norm_(model.parameters(), 1.)

                optimize_counter += 1

                if optimize_counter >= 150 * 16:
                    optimizer.step()
                    optimizer.zero_grad()

                    optimize_counter = 0

        optimizer.step()
        optimizer.zero_grad()

        if inferencehook:
            inferencehook(dataset, model, tfidf_model, num_sentences=num_sentences, k=10)

        this_epoch_loss = sum(epoch_losses) * optimize_every / len(epoch_losses)
        if last_saved_epoch_loss is None or this_epoch_loss < last_saved_epoch_loss:
            print('Saving {} with loss {:.8f}'.format(model_filename, this_epoch_loss))
            torch.save(model, 'modeldata/' + model_filename.format(loss=this_epoch_loss))
            last_saved_epoch_loss = this_epoch_loss
        epoch_losses = []


if __name__ == '__main__':
    train(model_filename='verse_continuation_model_{loss:.8f}.pt', inferencehook=generatorhook,
          lr=6.5e-5, correct_bias=False, epochs=1000, optimize_every=300,
          num_sentences=2, load_model='')
