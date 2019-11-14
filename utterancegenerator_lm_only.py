import torch  # pip3 install torch
from torch.nn import functional as F


def finish_utterance(gpt2_model, tfidf_model, tokenizer, verse, words2add=120, k=40):
    beal_start = tfidf_model.forward([verse])
    beal_start_sequence = tokenizer.encode(beal_start[0])
    new_sequence = []
    gpt2_model.eval()
    with torch.no_grad():
        for i in range(words2add):
            if len(beal_start_sequence) + len(new_sequence) >= 1024:
                print('reached max inference length')
                break
            predictions = gpt2_model(torch.Tensor([beal_start_sequence + new_sequence]).long().to('cuda'))
            prediction = predictions[0]
            topk = torch.topk(prediction, k)
            values = F.softmax(topk.values, dim=0)
            indices = topk.indices

            indices_index = torch.multinomial(values, 1).item()
            next_word_index = indices[indices_index].item()

            new_sequence.append(next_word_index)
    sentence = tokenizer.decode(new_sequence)
    gpt2_model.train()

    return sentence
