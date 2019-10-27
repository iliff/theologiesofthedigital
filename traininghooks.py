from utteranceassessor import predict_understanding
from utterancegenerator import finish_utterance


def generatorhook(dataset, model, sentences=[]):
    for start_sentence in sentences:
        p = finish_utterance(model, dataset.tokenizer, start_sentence, words2add=35)
        print(p)


def assessorhook(dataset, model, sentences=[]):
    for sentence in sentences:
        p = predict_understanding(model, dataset.tokenizer, sentence)
        print(p)
