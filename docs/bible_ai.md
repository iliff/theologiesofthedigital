# AI bible interpreter

## Rationale

Begin with bible as interface. As the materiality of bible moves beyond the dominance of the codex, do theologies of bible need to change? I have argued that considering bible as interface provides at least two advantages. first, bible as interface reminds us of the materiality of bible. bible can not be reduced simply to the content it contains. borrowing from its own etymology as a word, bible has always been material and REMAINS so in our increasingly digital age. second, thanks to Drucker's wonderful work, interface brings into focus the role of the user in constructing bible. In interface, users can not be reduced to simply consumers of content. Instead, users are participants in constructing bible as interface. maybe a third, interface highlights difference???
Our team believes that machines can teach us about difference. Reading with machines can impact the way we read. So, why not partner with machines to read bible and see what we learn. There are many ways to do this, such as different natural language processing techniques such as topic modeling or using page rank to find the most relevant passages in the bible or using markov techniques to build a twitter bot. 
In this project, we are building a text generator that will take a bible passage as an input and provide a reading/interpretation of that passage as an output. This is similar to a question/response engine. We could use similar techniques to build a bible generator that would write bible passages beginning with a prompt. This would be a more sophisticated version of the KJV bot and would be similar to "fake scripture." Yet, how do we decide it is fake?

> It might be interesting to have the model produce some passages and randomly show participants some of these and some "actual" passages and see if they can discern the difference and how? 

## Methodology

Do we want to use gpt-2 here or should we use Alberta or something better at call/response? 
* language model (link to papers discussing these things)
* corpora used for training
* python 
* compute setup
* interface design

## Questions

## Challenges


What kind of interface would I build? 

How could we make available for others to build upon? 

* What corpora would we want to use to tune gpt-2 to biblical interpretation? 
* How could we provide diverse perspectives? 
* How could users help train the model? 
* What happens if we feed the generated text back into the model?
* Is something as simple as daily bible reading useful? 
* Could this be used pedagogically to get converations started? 
* show the difference in result between untrained gpt-2 and trained gpt-2
* perhaps build a few different models with different corpora for tuning to see how things shift? 

Justin and Micah,

As usual, I have changed my mind on a project late in the game. On October 31, I have to submit my contribution to a Theologies of the Digital workshop that will happen in November. I was going to have the participants engage bible as API but i found that approach led me to just more writing and rewriting of the disseration bits for this audience. That is stupid. 
So, would you two be willing to help me (in a week) build a bible interpreter using gpt-2? User puts in a passage and model gives a few paragraphs of analysis. I know it is short notice and I will make time to do the work, but I would need your help in order to make this possible. 
If you are willing, 1) Justin, I would need to learn how to train on top of gpt-2 to tailor a model to a discourse and 2) Micah, I would need some help deciding on appropriate corpora to include in this additional training of the model.
I know this sounds like a corny project, but it will be much more beneficial to me to spend time doing this instead of writing a bunch of useless bullshit about interface and I think it can easily raise interesting questions about a theology of bible in the age of AI. 
I just have to have something to send them to engage on Oct 31. The model does not have to be finished. In fact, on Oct 31, I can send them the written justification for the project and the methodology along with some prelimiary reading on language models and text generation. And, if possible, I can give them a rudimentary interface to interact with gpt-2 tuned toward biblical interpretation? 
I should have started this weeks ago, but alas, I am a slacker. What do you think? Possible? Interested? 

Michael 