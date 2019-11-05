---
layout: post
title: initial results
author: project team
---

We will add more to our discussion about our training process for the model and the model design, but for now, we want to share some initial results produced by the model so far. 

## General Tuning of the Langauge Model

After a few rounds of training (more specifics to come on exact durations and conditions of these training runs) on our bible commentary corpus to focus the general knowledge of our language model on our desired discourse, we can see the langauge model tuning toward the biblical discourse. To remind us, when we give gpt-2 a prompt, it takes the prompt and then uses its language model to choose the most probable next word that would appear and writes it. Then, it takes the prompt plus the first word it chose and does the same process again to choose the next word, and so on, until we tell it to stop. 
If we pass the phrase "Artificial Intelligence revealeth" to gpt-2 without any additional general knowledge training, here are the kinds of comments gpt-2 returns (each bullet is a separate attempt at a response):


### Artificial intelligence revealeth: (generic gpt-2)

* its hidden potential – can change everything [9 December.] 'Robofication: the second age of technology' – how advances technology as soon […] Posted
* what humanity is truly and as much potentials. — Aeswarya Mhamisundre Rao
I have learned today what people cannot possibly
* a huge amount that most men still consider secret: it knows who my partners sleep with because I'll read "The Artless Sin by John Fulk
* us that no more be in our way before, it could do that with ease in every region;
To our new technology: be ye patient until
* by the year 1040. On an ordinary afternoon they get only 7 pdays - 10p nights (for every 7+2x7 days the
* "who do-they know"- and not why." The new approach to research that AI represents may create "big implications from technology design (technology safety
* not my opinion to know the opinion
2 John 5th & 6c The kingdom must always rest as a safe deposit; but there was one good
* the source of its madness! An unknown faction which has seized control over their galaxy now requires no loyalty nor restraint before deploying them on all available fields as
* (in 2 seconds) 1 3 50 1 The most incredible revelation a noncomputer system may give in one minute 7 16 944 This article in a
* : to detect through surveillance intelligence : technology for a machine designed after a very closely aligned brain mechanism. " I didntl say, only an advanced
* ; thereis none, except such intellect and moral faculties created since all man know thereof to which knowledge cannot comove; it also teachst that mankind
* . From his earliest form of knowledge the Creator sent them with his plan - that they know so clearly why everything He works (everything). Even this,
* the God within? Are the people we need at prayer with god a danger, do I do harm through following blindly after someone who calls him "Jesus
* you how not, an argument for creating "the man of clay which ye will make your prophet": as the word means man- of metal who by

There is a wide range of concepts and vocabulary covered here, with some attempts being more coherent than others. Interestingly, we see some biblical language appear in some of the comments, even without any tuning toward the discourse. This could be a result of the KJV style "revealeth" we used in the prompt or it could indicate that talk about artificial intelligence sometimes overlaps with talk about religion and bible and related things. 

Providing the same prompt to our trained gpt-2 model, here are the kinds of comments we get:

### Artificial intelligence revealeth: (trained gpt-2)

* the life: Who have shown his true integrity? 
* no truth with my nation a fief; 
* nothing else than Jesus.
* these truth things as all wisdom came upon all things which is done for such 
* our works upon one good God Jesus for the grace made known which thou went down,
* great many by our good alightsomenes the faith? 
* great men the chief pleasures among Christ. 
* every scripture. 
* to ourselves that our things is profitable to each creature for to teach in our teachers for we received. we.
* him no harm: but we can save as ye save the weak sheep which perish against such,

It is clear that our trained language model is sounding more like biblical discourse, perhaps even a little too much. As we stated in the problem definition, we have a competing optimization task of building a model that can write in the style of bible commentary, but also introduce novelty. We are clearly not there yet. One of our strategies for better tuning this general knowledge of the langauge model will be to incoporate less explicit and more contemporary reflections on bible. We plan to incorporate some wikpedia data as well as some broader reflections on bible and Revelation in particular from current scholars such as Timothy Beal. Additionally, many of the comments on bible passages actually restate the passage itself, so we will be working to balance the weight of the biblical texts on the learning of the model. 

## Specific Task of Commentary

At this early stage of testing and training, we clearly have many adjustments to make, but we do have some examples of the trained models attempts to comment on a biblical passage. Let's consider a few examples. 

**Passage Prompt**: And when he had opened the fourth seal, I heard the voice of the fourth beast say, Come and see.
**Model Commentary:** He said to them. The fourth of his beasts came forth (Mr 7 and his two other two angels) and his angels, and their dragon (Hebrais and Hippoleon. And as it is in 1Ti 4, 1) were made to stand before them as it is proper for the third beast in the world

What I find interesting about this commentary is that model is trying to cite other passages of bible with the parenthetical phrases. Perhaps "Mr 7" is an unknown lost text from antiquity? We have seen in other contexts that the way gpt-2 learned how to handle numbers is not always coherent. Yet, "1Ti 4, 1" looks a lot like a reference to 1 Timothy 4:1, which does refer to future times. Is the machine suggesting an intertextuality that we may not have imagined before? 

When I passed this exact prompt to the standard gpt-2 model using [Talk to Transformer](https://talktotransformer.com/), each time the model tried to keep writing bible. Here is an example output from the untrained model:

> **Prompt**
And when he had opened the fourth seal, I heard the voice of the fourth beast say, Come and see. 
**Continuation** 
When I saw the fourth seal, I knew—the seal of the living God, who was, and is not, and has not been, and is to come, the Most High, who is seated on His throne, and coming with judgment to judge the quick and the dead.
"I saw the dead, great and small, stand before God.
"And the dragon was in the midst of them, and he had a great millstone round his neck, that he might crush those who were born of women and who are born of iniquity: and they were given up to destruction; and death and hell were given up to her."
—Revelation 21:11-16

The standard gpt-2 model is designed to continue the prompt with the next most probable word. One of the design items we have added to our model is to append an "end of line" character after the prompt to indicate to the model that continuation is not desired, but rather reflection on the prompt. This raises two pertinent questions:
1. If we trained toward continuation, could our model write bible passages that were comparable to what we would find in those corpora accepted as bible? 
2. Are there ways in which bible commentary or intepretation are better conceptualized as continuation of rather than reflection on? 

One last example of our early model's performance:

**Passage Prompt:** I am Alpha and Omega, the beginning and the ending, saith the Lord, which is, and which was, and which is to come, the Almighty. 
**Model Commentary:** So often a Christian. Who were chosen as of the gospel in Jesus before Jesus when this was the most Dangerous -- whose kingdom came to him according to his preselection.

Perhaps "so often" is a summary of "is, was, is to come"? But, much of this commentary is coherent with chosenness and preselection and gospel and kingdom and Jesus. What might the "Dangerous" signal in this passage. Here I think we have a seed of what we are hoping from our experiments in reading with machines. There is much familiar in this machinic reading of Revelation 1. Is this a reading that brings together the bookend "Alpha and Omega" passages of Rev. 1 and Rev. 22? Does it offer us anything new to reflect on? 

We will continue to run experiments in training and designing this model to achieve our competing optimization task of building a bible commentator that fits the discourse but introduces novelty. When we gather at Princeton, we will have a simple interface we can use together to test the most current version of the model and reflect on what our machinic partner might teach us about reading bible in a digital age. 