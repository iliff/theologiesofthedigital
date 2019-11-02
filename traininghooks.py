from utterancegenerator import finish_utterance


def generatorhook(dataset, gpt2_model, tfidf_model, sentences=[], words2add=120):
    for start_sentence in sentences:
        p = finish_utterance(gpt2_model, tfidf_model, dataset.tokenizer, start_sentence,
                             words2add=words2add)
        print(p)
