import os

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
          sample_sentences=[], optimize_every=32):

    dataset = BibleCommentaryDataset(max_seq_len=512, max_dataset_length=500,
                                     batches_per_sent_len=4, df_book='Revelation')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=1)

    tfidf_model = CPULinear(num_output_sentences=1, knowledge_utterances=dataset.df.comment.tolist())

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

        dataset.set_sentence_length(epoch_i % 512)
        dataset.set_current_sample()

        running_losses = []

        for i, batch in enumerate(dataloader):

            X_gpt2, X_tfidf, y = batch

            # convert X_tfidf to sequences
            X_tfidf = tfidf_model.forward(X_tfidf)
            sequenced_X_tfidf = torch.Tensor([(dataset.tokenizer.encode(x) + [0] * 200)[:200] for x in X_tfidf]).long()

            # push X and y to cuda
            X_gpt2 = X_gpt2.to('cuda')
            X_tfidf = sequenced_X_tfidf.to('cuda')
            y = y.to('cuda')

            predictions = model(X_gpt2, X_tfidf)
            loss = criterion(predictions, y) / optimize_every
            running_losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            if loss % optimize_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                print('EPOCH {}, current_sentence_length {}, Batch {}: loss == {:.8f}'
                      .format(epoch, dataset.sentence_length, i,
                              sum(running_losses) * optimize_every / len(running_losses)))
                running_losses = []

        else:
            optimizer.step()
            optimizer.zero_grad()
            print('EPOCH {}, current_sentence_length {}, Batch {}: loss == {:.8f}'
                  .format(epoch, dataset.sentence_length, i,
                          sum(running_losses) * optimize_every / len(running_losses)))
            running_losses = []

        if inferencehook:
            sample_sentences = [s + ' ' + dataset.tokenizer.eos_token for s in sample_sentences]
            inferencehook(dataset, model, tfidf_model, sentences=sample_sentences)
        if last_loss is None and 'loss' in locals():
            last_loss = loss
        elif 'loss' in locals() and last_loss > loss:
            print('Saving {} with loss {:.8f}'.format(model_filename, loss))
            torch.save(model, 'modeldata/' + model_filename)
            last_loss = loss


if __name__ == '__main__':
    train(model_filename='verse_continuation_model.pt', inferencehook=generatorhook,
          lr=6.5e-5, correct_bias=False, epochs=1000, optimize_every=32,
          sample_sentences=['And they had hair as the hair of women, and their teeth were as the teeth of lions. ',
                            'And I saw no temple therein: for the Lord God Almighty and the Lamb are the temple of it. '])
