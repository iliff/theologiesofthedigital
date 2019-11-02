import torch  # pip3 install torch


def finish_utterance(gpt2_model, tfidf_model, tokenizer, start_of_sentence,
                     words2add=20):

    x_tfidf = tfidf_model.forward([start_of_sentence])
    # print('Most closely related sentence to the following:', x_tfidf)
    sequenced_x_tfidf = torch.Tensor([(tokenizer.encode(x) + [0] * 200)[:200] for x in x_tfidf]).long()

    sequence = tokenizer.encode(start_of_sentence)
    gpt2_model.eval()
    with torch.no_grad():
        for i in range(words2add):
            prediction = torch.argmax(gpt2_model(torch.Tensor([sequence]).long().to('cuda'),
                                                 sequenced_x_tfidf.to('cuda'))).item()
            sequence.append(prediction)
    sentence = tokenizer.decode(sequence)
    gpt2_model.train()

    return sentence
