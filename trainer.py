# TODO: change for this bible project!

import os

import torch  # pip3 install torch
# pip3 install pytorch-transformers
from pytorch_transformers import AdamW
from torch import nn
from torch.utils.data import DataLoader


def train(model_class, dataset_class, *dataset_args, model_filename='model.pt',
          lr=6.5e-5, correct_bias=False, epochs=1000, inferencehook=None,
          sample_sentences=[], **dataset_kwargs):
    dataset = dataset_class(*dataset_args, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True,
                            num_workers=1)

    if model_filename and os.path.exists(model_filename):
        print('loading model ...')
        model = torch.load(model_filename)
    else:
        model = model_class()
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    last_loss = None

    for epoch_i, epoch in enumerate(range(epochs)):

        running_losses = []
        for j in range(5, 25):  # already did 2 and 3, but then error
            for i, batch in enumerate(dataloader):

                optimizer.zero_grad()

                X, y = batch

                # push X and y to cuda
                X = X.to('cuda')
                y = y.to('cuda')

                predictions = model(X)
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

            dataset.set_sentence_length(j)
            dataloader = DataLoader(dataset, batch_size=4000 // j, shuffle=True,
                                    num_workers=1)
