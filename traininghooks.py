from utterancegenerator import finish_utterance


def generatorhook(dataset, gpt2_model, tfidf_model, sentences=[]):
    for start_sentence in sentences:
        p = finish_utterance(gpt2_model, tfidf_model, dataset.tokenizer, start_sentence,
                             words2add=20)
        print(p)
