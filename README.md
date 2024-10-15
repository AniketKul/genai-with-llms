# Generative AI with Large Language Models

Reference: https://www.deeplearning.ai/courses/generative-ai-with-llms

## Course Week 1

- Foundation models like gpt, llama, bert, flan-t5, bloom etc.
- LLM are able to take take prompts
- Context window of prompt is generation 1000 words
- The prompt is passed to the model, the model then predicts the next words, and because your prompt contained a question, this model generates an answer. The output of the model is called a completion, and the act of using the model to generate text is known as inference

- LLM use cases and tasks

An area of active development is augmenting LLMs by connecting them to external data sources or using them to invoke external APIs. You can use this ability to provide the model with information it doesn't know from its pre-training and to enable your model to power interactions with the real-world.

- Text generation before transformers

Previous generations of language models made use of an architecture called recurrent neural networks or RNNs. RNNs while powerful for their time, were limited by the amount of compute and memory needed to perform well at generative tasks

To successfully predict the next word, models need to see more than just the previous few words. Models needs to have an understanding of the whole sentence or even the whole document. 

Well in 2017, after the publication of this paper, Attention is All You Need, from Google and the University of Toronto, everything changed. The transformer architecture had arrived. This novel approach unlocked the progress in generative AI that we see today. It can be scaled efficiently to use multi-core GPUs, it can parallel process input data, making use of much larger training datasets, and crucially, it's able to learn to pay attention to the meaning of the words it's processing

### Transformers architecture

The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. To apply attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input. 

These attention weights are learned during LLM training and you'll learn more about this later this week. This diagram is called an attention map and can be useful to illustrate the attention weights between each word and every other word. Here in this stylized example, you can see that the word book is strongly connected with or paying attention to the word teacher and the word student. This is called self-attention and the ability to learn a tension in this way across the whole input significantly approves the model's ability to encode language

The transformer architecture is split into two distinct parts, the encoder and the decoder. These components work in conjunction with each other and they share a number of similarities. Also, note here, the diagram you see is derived from the original attention is all you need paper. Notice how the inputs to the model are at the bottom and the outputs are at the top, where possible we'll try to remain faithful to this throughout the course

Now, machine-learning models are just big statistical calculators and they work with numbers, not words. So before passing texts into the model to process, you must first tokenize the words. Simply put, this converts the words into numbers, with each number representing a position in a dictionary of all the possible words that the model can work with. You can choose from multiple tokenization methods.

What's important is that once you've selected a tokenizer to train the model, you must use the same tokenizer when you generate text. Now that your input is represented as numbers, you can pass it to the embedding layer. 

This layer is a trainable vector embedding space, a high-dimensional space where each token is represented as a vector and occupies a unique location within that space. Each token ID in the vocabulary is matched to a multi-dimensional vector, and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence. Embedding vector spaces have been used in natural language processing for some time, previous generation language algorithms like Word2vec use this concept

word -> token id -> embedding vector

In original transformers paper, the vector size was 512

For simplicity, if you imagine a vector size of just three, you could plot the words into a three-dimensional space and see the relationships between those words. You can see now how you can relate words that are located close to each other in the embedding space, and how you can calculate the distance between the words as an angle, which gives the model the ability to mathematically understand language.

As you add the token vectors into the base of the encoder or the decoder, you also add positional encoding. The model processes each of the input tokens in parallel. So by adding the positional encoding, you preserve the information about the word order and don't lose the relevance of the position of the word in the sentence.

Once you've summed the input tokens and the positional encodings, you pass the resulting vectors to the self-attention layer. Here, the model analyzes the relationships between the tokens in your input sequence. As you saw earlier, this allows the model to attend to different parts of the input sequence to better capture the contextual dependencies between the words. The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence

But this does not happen just once, the transformer architecture actually has multi-headed self-attention. This means that multiple sets of self-attention weights or heads are learned in parallel independently of each other. The number of attention heads included in the attention layer varies from model to model, but numbers in the range of 12-100 are common.

The intuition here is that each self-attention head will learn a different aspect of language. For example, one head may see the relationship between the people entities in our sentence. Whilst another head may focus on the activity of the sentence. Whilst yet another head may focus on some other properties such as if the words rhyme. 

It's important to note that you don't dictate ahead of time what aspects of language the attention heads will learn. The weights of each head are randomly initialized and given sufficient training data and time, each will learn different aspects of language

Now that all of the attention weights have been applied to your input data, the output is processed through a fully-connected feed-forward network. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary. You can then pass these logits to a final softmax layer, where they are normalized into a probability score for each word. This output includes a probability for every single word in the vocabulary, so there's likely to be thousands of scores here. One single token will have a score higher than the rest. This is the most likely predicted token. But as you'll see later in the course, there are a number of methods that you can use to vary the final selection from this vector of probabilities.

### Generating text with transformers

- Let's walk through a simple example. In this example, you'll look at a translation task or a sequence-to-sequence task, which incidentally was the original objective of the transformer architecture designers. You'll use a transformer model to translate the French phrase [FOREIGN] into English.

- First, you'll tokenize the input words using this same tokenizer that was used to train the network. These tokens are then added into the input on the encoder side of the network, passed through the embedding layer, and then fed into the multi-headed attention layers. The outputs of the multi-headed attention layers are fed through a feed-forward network to the output of the encoder. At this point, the data that leaves the encoder is a deep representation of the structure and meaning of the input sequence
- This representation is inserted into the middle of the decoder to influence the decoder's self-attention mechanisms
- Next, a start of sequence token is added to the input of the decoder. This triggers the decoder to predict the next token, which it does based on the contextual understanding that it's being provided from the encoder
- The output of the decoder's self-attention layers gets passed through the decoder feed-forward network and through a final softmax output layer. At this point, we have our first token. You'll continue this loop, passing the output token back to the input to trigger the generation of the next token, until the model predicts an end-of-sequence token. At this point, the final sequence of tokens can be detokenized into words, and you have your output.
- There are multiple ways in which you can use the output from the softmax layer to predict the next token. These can influence how creative you are generated text is.

- 
Encoder-only models also work as sequence-to-sequence models, but without further modification, the input sequence and the output sequence or the same length. Their use is less common these days, but by adding additional layers to the architecture, you can train encoder-only models to perform classification tasks such as sentiment analysis, BERT is an example of an encoder-only model.
- Examples of encoder-decoder models include BART as opposed to BERT and T5
- Finally, decoder-only models are some of the most commonly used today. Again, as they have scaled, their capabilities have grown. These models can now generalize to most tasks. Popular decoder-only models include the GPT family of models, BLOOM, Jurassic, LLaMA, and many more


### Prompting and Prompt Engineering

- The text that you feed into the model is called the prompt, the act of generating text is known as inference, and the output text is known as the completion. The full amount of text or the memory that is available to use for the prompt is called the context window
- You may have to revise the language in your prompt or the way that it's written several times to get the model to behave in the way that you want. This work to develop and improve the prompt is known as prompt engineering


### Generative Configuration

- You'll examine some of the methods and associated configuration parameters that you can use to influence the way that the model makes the final decision about next-word generation.
- If you've used LLMs in playgrounds such as on the Hugging Face website or an AWS, you might have been presented with controls like these to adjust how the LLM behaves.
- Note that these are different than training parameters which are learned during training time
- Instead, these configuration parameters are invoked at inference time and give you control over things like the maximum number of tokens in the completion, and how creative the output is (temperature).
-  
The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses. Here you can see a selection of words and their probability score next to them.
- Most large language models by default will operate with so-called greedy decoding. This is the simplest form of next-word prediction, where the model will always choose the word with the highest probability. 
- If you want to generate text that's more natural, more creative and avoids repeating words, you need to use some other controls. Random sampling is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection.
- However, depending on the setting, there is a possibility that the output may be too creative, producing words that cause the generation to wander off into topics or words that just don't make sense. Note that in some implementations, you may need to disable greedy and enable random sampling explicitly. For example, the Hugging Face transformers implementation that we use in the lab requires that we set do sample to equal true
- 
Two Settings, top p and top k are sampling techniques that we can use to help limit the random sampling and increase the chance that the output will be sensible. To limit the options while still allowing some variability, you can specify a top k value which instructs the model to choose from only the k tokens with the highest probability.

- Alternatively, you can use the top p setting to limit the random sampling to the predictions whose combined probabilities do not exceed p. For example, if you set p to equal 0.3, the options are cake and donut since their probabilities of 0.2 and 0.1 add up to 0.3. The model then uses the random probability weighting method to choose from these tokens

- With top k, you specify the number of tokens to randomly choose from, and with top p, you specify the total probability that you want the model to choose from

- This parameter influences the shape of the probability distribution that the model calculates for the next token. Broadly speaking, the higher the temperature, the higher the randomness, and the lower the temperature, the lower the randomness. The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability distribution of the next token. In contrast to the top k and top p parameters, changing the temperature actually alters the predictions that the model will make. If you choose a low value of temperature, say less than one, the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words

- If you leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used. 


### Generative AI project lifecycle

- Scope: define the use case
- Select: choose an existing model or pretrain your own
- Adapt and Align Model: Prompt engineering, fine-tuning, align with human feedback and evaluate
- Application Integration: Optimize and deploy model for inference, Augment model and build LLM-powered applications




## Course Week 2

## Course Week 3