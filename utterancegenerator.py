import torch  # pip3 install torch


def finish_utterance(gpt2_model, tfidf_model, tokenizer, verse,
                     words2add=20, k=40):

    x_tfidf = tfidf_model.forward([verse])
    # print('Most closely related sentence to the following:', x_tfidf)
    sequenced_x_tfidf = torch.Tensor([(tokenizer.encode(x) + [0] * 200)[:200] for x in x_tfidf]).long()

    eos_index = tokenizer.encode(tokenizer.eos_token)[0]
    new_sentence = [eos_index]

    sequence = tokenizer.encode(verse)
    gpt2_model.eval()
    with torch.no_grad():
        for i in range(words2add):
            if len(sequence) >= 1024:
                print('reached max inference length')
                break
            predictions = gpt2_model(torch.Tensor([sequence]).long().to('cuda'), sequenced_x_tfidf.to('cuda'),
                                     torch.Tensor([new_sentence]).long().to('cuda'))
            prediction = predictions[0]
            top_k = torch.topk(prediction, k)
            p = top_k.values
            p_index = torch.multinomial(p, 1).item()
            new_sentence.append(top_k.indices[p_index].item())
    sentence = tokenizer.decode(new_sentence)
    sentence = sentence[len(verse):]
    gpt2_model.train()

    return sentence
