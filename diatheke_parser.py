#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import json

SCRIPTURE_REFERENCES = re.compile(r'^([a-zA-Z ]+ \d+:\d+):', re.M)
with open('../scripture_dict.json') as fp:
    SCRIPTURE_DICT = json.load(fp)


def diatheke(scripture, work="KJVA"):
    cmd = ['diatheke', '-b', work, '-f', 'plain', '-k', scripture]
    text_object = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE)
    citation_text = []
    full_output = '\n'.join(text_object.stdout.readlines())
    text_list = SCRIPTURE_REFERENCES.split(full_output)
    '''
    if text_list[0] == '':
        del text_list[0]
    '''
    text_list = [x for x in text_list if x not in ('', None)]
    citation_text = []
    for index in range(0,len(text_list)):
        if index % 2 == 0:
            try:
                citation_text.append([text_list[index].strip(), text_list[index + 1].replace('\n', '\t')])
            except IndexError:
                pass

    return citation_text


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-b', '--book', help="Sword book to investigate")
    aparser.add_argument('-c', '--commentary', help="Mark if the work is a commentary", action='store_true')
    aparser.add_argument('scripture', help='Scripture citation to return results from')
    args = aparser.parse_args()

    if args.book:
        work = args.book
    else:
        work = 'KJVA'

    citation_text = diatheke(args.scripture, work)
    if args.commentary:
        citation_scripture_comments = []
        for line in citation_text:
            try:
                text = SCRIPTURE_DICT[line[0]]
            except IndexError:
                print('line 61')
                print(line)
                pass
            citation_scripture_comments.append([line[0], text, line[1]])
        citation_text = citation_scripture_comments 

    outputfile = work + '.tsv'
    with open(outputfile, 'w+') as fp:
        for line in citation_text:
            fp.write('|'.join(line) + '\n')


    
if __name__ == '__main__':
    import argparse

    main()
