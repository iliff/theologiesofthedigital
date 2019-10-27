import torch  # pip3 install torch


def finish_utterance(model, tokenizer, start_of_sentence, words2add=20):
    sequence = tokenizer.encode(start_of_sentence)
    model.eval()
    with torch.no_grad():
        for i in range(words2add):
            prediction = torch.argmax(model(torch.Tensor([sequence]).long().to('cuda'))).item()
            sequence.append(prediction)
    sentence = tokenizer.decode(sequence)
    model.train()

    return sentence
