import re

import torch
from pytorch_transformers import GPT2Tokenizer

from datasets.datasets_lm_only import BibleCommentaryDataset
from models.generator_lm_only import CPULinear, GPT2Generator
from utterancegenerator_lm_only import finish_utterance


def infer_conditionally(model, tfidf_model, tokenizer, remove_duplicate_phrases=True):
    while True:
        verse = input('Please enter your verse: ')

        words2add = input('How many words should I add? ')
        try:
            words2add = int(words2add)
        except ValueError:
            print(words2add, 'is not an integer')
            continue

        top_k = input('How many words should we choose from at each step (token/word)? ')
        try:
            top_k = int(top_k)
        except ValueError:
            print(top_k, 'is not an integer')
            continue

        variations = 1
        if top_k > 1:
            variations = input('How many variations of commentary for this verse would you like to see? ')
            try:
                variations = int(variations)
            except ValueError:
                print(variations, 'is not an integer')
                continue

        for i in range(variations):
            # gpt2_model, tfidf_model, tokenizer, verse, words2add=120, k=40
            utterance = finish_utterance(model, tfidf_model, tokenizer, verse, words2add=words2add, k=top_k)
            # start at the beginning of a sentence
            if re.search(r'[?.!]\s+(?=[A-Z])', utterance):
                utterance = re.split(r'[?.!]\s+(?=[A-Z])', utterance, maxsplit=1)[1]
            if remove_duplicate_phrases:

                # remove single duplicate words
                utterance = re.sub(r'([\w.,;:!?"]+ )\1{1,9}', r'\1', utterance)

                # remove multiple duplicate words
                for j in range(9, 1, -1):
                    pattern = r'([\w.,;:!?"]+ |$)' * j
                    repeats = r'(' + r''.join([r'(\{})'.format(k) for k in range(1, j + 1)]) + r')+'
                    utterance = re.sub(pattern + repeats, r'\{}'.format(j + 1), utterance + ' ')

            utterance = utterance.strip().replace('\n', ' ')

            print('INFERENCE {} >'.format(i + 1), utterance + '\n')


if __name__ == '__main__':
    model = torch.load('../modeldata/model_0.04164963.pt').to('cuda')
    # model = GPT2Generator().to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    dataset = BibleCommentaryDataset(dir_='../trainingdata', filenames=['Beal.txt'], min_sequence_length=10,
                                     max_sequence_length=300)
    dataset.current_sequence_length = 30
    tfidf_model = CPULinear(output_sent_indices_to_join=[0, 1],
                            knowledge_utterances=[dataset.tokenizer.decode(dataset[i][0].tolist()) for i in
                                                  range(len(dataset))])

    infer_conditionally(model, tfidf_model, tokenizer)
