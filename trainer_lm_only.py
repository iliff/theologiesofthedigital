import os

import torch  # pip3 install torch
# pip3 install pytorch-transformers
from pytorch_transformers import AdamW
from torch import nn
from torch.utils.data import DataLoader

from datasets.datasets_lm_only import BibleCommentaryDataset
from models.generator_lm_only import CPULinear, GPT2Generator
from traininghooks_lm_only import generatorhook


def train(model_filename='model_{loss}.pt', lr=6.5e-5, epochs=1000, inferencehook=None,
          load_model=None, inference_verses=2, batch_size=64):
    dataset = BibleCommentaryDataset(dir_='trainingdata', filenames=['Beal.txt'], min_sequence_length=10,
                                     max_sequence_length=300)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=1)

    # creates tfidf model ONLY ON SEQUENCE SIZE 30 (for now)
    dataset.current_sequence_length = 30
    tfidf_model = CPULinear(output_sent_indices_to_join=[1, 3],
                            knowledge_utterances=[dataset.tokenizer.decode(dataset[i][0].tolist()) for i in range(len(dataset))])
    dataset.current_sequence_length = dataset.min_sequence_length

    if load_model:
        print('loading model {} ...'.format(load_model))
        model = torch.load(os.path.join('modeldata', load_model))
    else:
        model = GPT2Generator()
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=False)

    optimizer.zero_grad()

    epoch_losses = []
    last_saved_epoch_loss = None

    for epoch_i, epoch in enumerate(range(epochs)):

        for i, batch in enumerate(dataloader):

            X, y = batch

            # push X and y to cuda
            X = X.to('cuda')
            y = y.to('cuda')

            predictions = model(X)
            loss = criterion(predictions, y)
            epoch_losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            optimizer.zero_grad()
            print('EPOCH {}, current_sequence_length {}, Batch {} of {}: loss == {:.8f}'
                  .format(epoch, dataset.current_sequence_length, i, (len(dataset) + 1) // batch_size, loss.item()))

            if inferencehook and i % 100 == 0:
                inferencehook(dataset, model, tfidf_model, inference_verses=inference_verses, words2add=150, k=20)

        this_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        if last_saved_epoch_loss is None or this_epoch_loss < last_saved_epoch_loss:
            print('Saving {} with loss {:.8f}'.format(model_filename, this_epoch_loss))
            torch.save(model, 'modeldata/' + model_filename.format(loss=this_epoch_loss))
            last_saved_epoch_loss = this_epoch_loss
        epoch_losses = []

        if dataset.current_sequence_length == dataset.max_sequence_length:
            dataset.current_sequence_length = dataset.min_sequence_length
        else:
            dataset.current_sequence_length += 1


if __name__ == '__main__':
    train(model_filename='model_{loss:.8f}.pt', inferencehook=generatorhook,
          lr=6.5e-5, epochs=1000, load_model='model_3.78957510.pt', batch_size=128,
          inference_verses=['Come, I will show you the judgment of the great whore who is seated on many waters, '
                            'with whom the kings of the earth have committed fornication, and with the wine of '
                            'whose fornication the inhabitants of the earth have become drunk.',
                            'Then I saw a new heaven and a new earth; for the first heaven and the first earth '
                            'had passed away, and the sea was no more.'])
