import numpy as np
from utterancegenerator_lm_only import finish_utterance


def generatorhook(dataset, gpt2_model, num_sentences=4, words2add=120, k=40):
    sentences = dataset.df[dataset.df['reference'].str.contains('Revelation')]['verse']
    for start_sentence in sentences.sample(n=num_sentences):
        p = finish_utterance(gpt2_model, dataset.tokenizer, start_sentence,
                             words2add=words2add, k=k)
        print('"' + start_sentence + '"', '=>', p)
