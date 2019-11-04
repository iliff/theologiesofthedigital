import torch  # pip3 install torch


def finish_utterance(gpt2_model, tokenizer, verse, words2add=20, k=40):

    sequence = tokenizer.encode(verse)
    gpt2_model.eval()
    with torch.no_grad():
        for i in range(words2add):
            if len(sequence) >= 1024:
                print('reached max inference length')
                break
            predictions = gpt2_model(torch.Tensor([sequence]).long().to('cuda'))
            prediction = predictions[0]
            top_k = torch.topk(prediction, k)
            p = top_k.values
            p_index = torch.multinomial(p, 1).item()
            sequence.append(top_k.indices[p_index].item())
    sentence = tokenizer.decode(sequence)
    sentence = sentence[len(verse):]
    gpt2_model.train()

    return sentence
