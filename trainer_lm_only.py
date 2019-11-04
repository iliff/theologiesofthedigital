import os

import torch  # pip3 install torch
# pip3 install pytorch-transformers
from pytorch_transformers import AdamW
from torch import nn
from torch.utils.data import DataLoader

from datasets.datasets_lm_only import BibleCommentaryDataset
from models.generator_lm_only import GPT2Generator
from traininghooks_lm_only import generatorhook


def train(model_filename='verse_continuation_model.pt',
          lr=6.5e-5, correct_bias=False, epochs=1000, inferencehook=None,
          num_sentences=4, optimize_every=1):
    dataset = BibleCommentaryDataset(max_seq_len=512, max_dataset_length=300,
                                     batches_per_sent_len=4, df_book='Revelation')
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True,
                            num_workers=1)

    if model_filename and os.path.exists(model_filename):
        print('loading model ...')
        model = torch.load(model_filename)
    else:
        model = GPT2Generator()
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    optimizer.zero_grad()

    last_loss = None

    for epoch_i, epoch in enumerate(range(epochs)):

        dataset.set_sentence_length(151)
        dataset.set_current_sample()

        running_losses = []

        for i, batch in enumerate(dataloader):

            X, y = batch

            # push X and y to cuda
            X = X.to('cuda')
            y = y.to('cuda')

            for j in range(1, 151):
                predictions = model(X[:, :j])
                loss = criterion(predictions, X[:, j] if j < 150 else y) / optimize_every
                running_losses.append(loss.item())
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.)

                if j % optimize_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print('EPOCH {}, current_sentence_length {}, Batch {}: loss == {:.8f}'
                          .format(epoch, j, i,
                                  sum(running_losses) * optimize_every / len(running_losses)))
                    running_losses = []

                if inferencehook and j % 50 == 0:
                    inferencehook(dataset, model, num_sentences=num_sentences)

            else:
                optimizer.step()
                optimizer.zero_grad()

        if last_loss is None and 'loss' in locals():
            last_loss = loss
        elif 'loss' in locals() and last_loss > loss:
            print('Saving {} with loss {:.8f}'.format(model_filename, loss))
            torch.save(model, 'modeldata/' + model_filename)
            last_loss = loss


if __name__ == '__main__':
    train(model_filename='verse_continuation_model_lm_only.pt', inferencehook=generatorhook,
          lr=6.5e-5, correct_bias=False, epochs=1000, optimize_every=4,
          num_sentences=6)
