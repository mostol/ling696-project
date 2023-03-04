# LING 696G Final Project
> 1. This is due March 6 by 12:00 (noon).
> 2. I'm looking for a "small" TTS or STT system. You will submit code and a brief writeup.
> 3. Your code should be carefully commented. Your comments should be sufficient so that I can run your code myself should I choose to.
> 4. Your submission needs to be more than just a system for some language. You must have at least two versions of your system where you tweak parameters, data, training and then evaluate the effect of that.
> 5. There needs to be a brief (less than 5pp) write-up that explains:
>    1. the general structure of your systems;
>    2. what variants you tested and why you thought they were worth testing;
>    3. and which one works better.

For my final project, I wanted to build a text-to-speech (TTS) system. Because I used VITS for the mid-course excercise and had heard about lower performance and longer training times for other models like Tacotron, I decided to look for a relatively performant alternative and settled on the [DelightfulTTS](https://arxiv.org/abs/2207.04646) model, which has has an [in-development branch](https://github.com/coqui-ai/TTS/tree/delightful-tts) in the Coqui TTS package. I trained the model on the Dutch language Common Voice dataset so that I could get an idea of the models' performance compared to my previous attempt.

## Model and overall approach
Like VITS, DelightfulTTS is an end-to-end text-to-speech model. The main distinguishing feature of DelightfulTTS is its focus on prosody and improving the performance of speech representation generation (in this case, mel-spectrograms) by generating down-sampled predictions which are then up-sampled by the vocoder. Because this approach creates a generalizable acoustic model and vocoder, explicit labels for speakers and languages can be supplied and the model can learn representations for individual speakers or specific languages while training on a large and varied dataset.

Since DelightfulTTS can leverage single-speaker information while training on a multi-speaker dataset, I decided to train two models on the full Dutch Common Voice dataset for comparison: one treating all speakers as a single speaker, and one that tracked speaker IDs and learned individual speaker embeddings. With my earlier VITS model, I managed to generate speech that was fairly convincing in tone and vocal quality by training on a single-speaker dataset; however, because of the limited amount of data, utterances were often indistinct and inference quickly degenerated when faces with situations like out-of-sample vocabulary or longer text inputs. A larger dataset could help resolve these issues, but without a way to single out speakers the result would likely be inhuman-sounding. With an expanded dataset combined with DelightfulTTS's speaker embedding capacities, I hoped that I could have a model that could produce robust Dutch speech while also generating audio that sounded as if it came from a single person rather than a cacophony of speakers at once.

## Parameters and training approaches
Since my primary point of comparison was intended to be single- vs. multi-speaker setups, I tried to keep other parameters consistent between the two models and to select optimal configurations:
* Rather than using a character-based configuration, I used phonemes by taking advantage of `gruut`'s Dutch phoneme selection. This required installing an extension to the `gruut` package.
* For the dataset, I used the `validated.tsv` Common Voice data directly and leveraged Coqui's built-in `common_voice` dataset formatter for convenience and consistency.
* To ensure ample opportunity to produce the best model possible, I configured training to run for 100 epochs With a large dataset, more epochs aren't always necessary—in fact, these models converged well before 100 epochs.

This required setting up a new container image:
<container image here>

## Results
(Needed to fork the repository to make some fixes before I could do inference)

The two models are essentially indistinguishable.
