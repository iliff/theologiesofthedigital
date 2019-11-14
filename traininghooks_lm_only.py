import torch
from utterancegenerator_lm_only import finish_utterance


def generatorhook(dataset, gpt2_model, tfidf_model, inference_verses=[], words2add=120, k=40):
    for verse in inference_verses:
        p = finish_utterance(gpt2_model, tfidf_model, dataset.tokenizer, verse,
                             words2add=words2add, k=k)
        print('"' + verse + '"', '=>', p)
