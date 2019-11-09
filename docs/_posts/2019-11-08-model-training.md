---
layout: post
title: model training
author: project team
---

We are taking a supervised approach to training this bible commentator model. This means that we begin with the gpt-2 pre-trained language model, which is a neural network with several layers depending on which size of the model we use. 
> Just this week, [OpenAI has released their X-Large gpt-2 model](https://openai.com/blog/gpt-2-1-5b-release/), which has 1.5 billion parameters and 48 layers. This piece by Jay Alammar, ["The Illustrated GPT-2"](http://jalammar.github.io/illustrated-gpt2/), provides some nice visualizations of the different sizes of the gpt-2 model. Since the X-Large has not been available, and due to the demand on compute resources and the design of our model, which trains 2 gpt-2 models in parallel, we have focused on the Large and Medium gpt-2 models, which have 32 and 24 layers respectively. 

With our supervised approach to training, we construct a dataset that has two inputs (X_verse, X_commentary) and a known output (y). Because we are using gpt-2 to generate text, one word at a time, from a seed prompt, here is what these inputs and outputs look like:

```
X_scripture = an integer encoded sequence of the entire verse from Revelation
X_commentary = an interger encoded sequence of the commentary at each stage (this will grow each pass)
y = the next word (encoded sequence) of the commentary 
```

Every pass through gpt-2, our model uses these X inputs to predict the next most likely word in the commentary and appends this word to the X_commentary input for the next training pass. Since this is supervised learning, this word that the model predicts (y^) is compared against what we know to be the next word (y) in the commentary we are using as our training dataset and a loss is calculated based on the difference between y (actual next word) and y^ (predicted next word). Using this loss calculation, the model goes backward through its layers and adjusts the weights of each connection in the network before it runs through the next pass. To make our training a bit more resonable to manage, we actually only adjust the weights of the edges of the network after approximately 16 commentary samples have been processed. 
> For an excellent vizualization of the intuition around this backward process of a neural network, where weights are recalculated, see 3Blue1Brown's excellent video https://youtu.be/Ilg3gGewQ5U. 

We have defined our commentary length output to be 151 words, so we ask the model to make this many predictions for each verse input. 

## Adding Knowledge to our Training
To provide more targeted and nuanced training for our model, we are experimenting with a more complicatd training process that passes an additional input to the model indicating a best fit generic background knowledge sequence for the commentary we are training on. So, in addition to X_verse and X_commentary, we will pass the model X_tfidf, an encoded sequence representing a passage from a more generic corpus related to Revelation. The reason we call this input X_tfidf is that we use a similarity based linear model to find relevant knowledge passages related to a given commentary sequence and then we use term-frequency-inverse-document-frequency (tf-idf) to pick the most informative passage from the list. For more details on this additional training aspect, see the CPULinear class in our generator.py file. 
We are hoping this additional knowledge input will provide better intuition for the model to generate text that fits the commentary discourse without over-fitting to that discourse. This is one strategy we are employing to push toward our dual optimization task of fitting the discourse and introducing novelty. 