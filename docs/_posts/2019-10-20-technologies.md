---
layout: post
title: technologies
author: project team
---

In the interest of being intentional about the materialities at work in this project and to foster other's reproducing and expanding on this work, we will define the main technologies we will use to build this project.

## Python 
[Python](https://www.python.org/) is a highly readable programming language particularly useful for data science and machine learning. As we have outlined in "Library as Interface for DH Work,"[^lib_dh] we have invested in Python as a core competency in our Experimental Humanities Lab for several reasons. The most important reasons for choosing Python for this particular porject are Python's extensive set of openly available libraries to work with natural language processing tasks such as data preparation and with cutting edge deep learning frameworks such as PyTorch and gpt-2 (see below). The focus on readability in Python also provides an excellent environment for teaching and collaboration, lowering the barriers for people to engage the code. 

[^lib_dh]: Experimental Humanities @ Iliff, "Library as Interface for DH Work," Clifford B. Anderson, ed., *Digital Humanities and Libraries and Archives in Religious Studies* (Berlin: De Gruyter, 2019).

## GPU 
One of the major advances in machine learning computation power over the past several years has been the growing accessibility of graphical processing units (GPU). Mythbusters provide a very useful enactment of the difference between a GPU and a more traditional central processing unit (CPU):

[![GPU vs. CPU](http://img.youtube.com/vi/-P28LKWTzrI/0.jpg)](http://www.youtube.com/watch?v=-P28LKWTzrI "GPU vs. CPU")

The main advantage of a GPU for machine learning tasks is the capability for parallel processing. Structures like neural networks with many layers and millions of parameters require a large amount of processing. Leveraging a GPU allows us to train models on large data sets much faster so we can iterate through experiments more quickly to optimize our models.[^gpu]

Thanks to generous funding from the [Henry Luce Foundation](https://www.hluce.org/programs/theology/), [Iliff's AI Institute](https://ai.iliff.edu) has a dedicated GPU server with an [NVidia Tesla V100 GPU](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf) and 32GB of memory that we are able to use to train our models for this project. 

[^gpu]: For more information on why a GPU can be useful in machine learning applications, there are many explanations available online. A good example is [Faizan Shaikh's "Why are GPUs necessary for training Deep Learning models?"](https://www.analyticsvidhya.com/blog/2017/05/gpus-necessary-for-deep-learning/)

## gpt-2
In the last few years, we have seen an explosion in the development of powerful pre-trained language models that can be used as a foundation for several natural language processing tasks, such as text generation, question answering, machine translation, and more. The two language models we use in our development currently are [BERT](https://arxiv.org/abs/1810.04805) from Google and [gpt-2](https://openai.com/blog/better-language-models/) from OpenAI. These language models provide a statistical representation of a language (we are currently focused on English) that can be tuned to a specifc discourse and then used for several tasks.[^embedding] 
For this bible commentator project, we are using gpt-2 to generate text one word at a time in response to a prompt verse. Trained on text from 1.5 million web pages top generate a generic language model, gpt-2 prevents us from having to build a language model from scratch on a much smaller dataset. Instead, we can build on top of the generic language model to tune the model toward our particular task by training gpt-2 using our particular bible commentary corpora. 

If you would like to see a demonstration of gpt-2 text generation in response to an input, you can visit [Adam King's simple web interface for gpt-2](https://talktotransformer.com/). Just enter a prompt and see what the model writes. 

[^embedding]: The vectorization of language used in the embedding techniques of these language models deserves far more attention than we can give it here. The ways machines can represent language through single or multi-dimensional matrices of numbers could be a distinct source of the difference that machines bring to the reading and writing task. We believe we can learn from machines here. 

## GitHub 
GitHub is a collaborative cloud code repository that allows our team to work together on the project and share with others. Using git revision control system along with GitHub provides granular access to every revision made to the project and makes it easy for several developers across wide geographies to contribute to the project. 

We also use GitHub as a platform for scholarly communications, leveraging the built in static site generator included with github. Building on another project from Iliff's Experimental Humanities Lab, we have used our [template for digital projects](https://github.com/iliff/digital-dissertation) on GitHub to scaffold the repository for this project. GitHub as a scholarly communication framework allows us to share our research and invite collaboration through the readability of our python code base itself, through careful commenting of the code, and through short pieces of web writing to reflect on the project and process (this page is an example of this last option). 