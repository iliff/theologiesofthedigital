import torch
from pytorch_transformers import GPT2Tokenizer

from utterancegenerator_lm_only import finish_utterance


if __name__ == '__main__':
    model = torch.load('../modeldata/verse_continuation_model_lm_only_2.79289041.pt').to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

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
            utterance = finish_utterance(model, tokenizer, verse, words2add=words2add, k=top_k)
            print(utterance)
