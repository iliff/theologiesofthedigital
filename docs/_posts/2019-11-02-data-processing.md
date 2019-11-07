---
layout: post
title: data processing
author: project team
---

## Loading Texts
As mentioned in our [data collection](./2019-10-22-data-collection.md) reflections, we have initially limited oursleves to task specific knowledge from the SWORD project. Part of the reason for this is that SWORD and Crosswire have made it easy to access their texts in machine readable fashion through a customizable command-line interface called [diatheke](https://wiki.crosswire.org/Frontends:Diatheke) and texts encoded in [OSIS XML](https://en.wikipedia.org/wiki/Open_Scripture_Information_Standard), one of the most common XML standards for texts related to bible.

> Arguably the largest digital collection of bible corpora in the world is the Digital Bible Library (DBL). Interestingly, the DBL has not opted to use OSIS as their XML standard. DBL uses an XML standard called [USX](https://ubsicap.github.io/usx/), which is based on the older USFM schema. We have not been able to get licensing from DBL at this time to use any of the corpora it contains beyond what is already available in public domain. We are hoping to get access to more DBL resources at some point. 

The standard XML format of the SWORD commentaries and the customizability of the command line interface allowed us to write a parser to read any SWORD commentary and split the text into 3 columns: bible citation, text of citation, text of commentary. See our [diatheke_parser.py file](https://github.com/iliff/theologiesofthedigital/blob/master/diatheke_parser.py) in the repository for this project to see the code for this parser. 

## Tokenizing
Computers work better with numbers than letters or words, so the next step of our data processing is to tokenize and encode all of the text we will pass to our model. Tokenizing simply means splitting a text up into smaller parts (i.e. tokens). Tokens can be letters, words, parts of words, punctuation, etc., depending on your task and the tokenizer you use. With some slight modifications, we use the GPT2Tokenizer to split up the verses and the commentary in our data into smaller bits that correpond to the 50,000 word vocabulary of gpt-2. Each item in this vocabulary has a corresponding integer value, so when we tokenize our text we also encode as integers.
Before we tokenize and encode, there is some cleaning up to do, such as removing unwanted characters, removing duplicate comments, and trimming out anything else in the data that will just add unwanted noise. This cleaning step may seem mundane, but it is often one of the places in a project where certain assumptions about the dataset get enacted. So, we need to be transparent about the task involved in the data preparation and always question them as we learn more about how the model uses the data. At present, here are the things we do to clean our dataset before passing it to the model:

```python
def _clean_df(self, df):
        df = df.dropna(subset=['comment'])
        df = df.drop_duplicates(subset=['comment'], keep='last')
        df.loc[:, 'comment'] = df['comment'].apply(lambda x: x.strip())
        df = df[df['comment'] != '']
        return df
```

## Preparing Data
Once our texts are tokenized and encoded, we construct a dataframe, which looks much like a spreadsheet. Our dataframe has one column with the encoded version of a verse from Revelation and one column with the corresponding encoded version of the commentary on that verse. We have as many rows in the dataframe as we have samples of verses and commentary from the commentaries we pulled from SWORD using the diatheke interface. We could add many more rows to this dataframe as we increase the commentary data sources we can incorporate into the model.