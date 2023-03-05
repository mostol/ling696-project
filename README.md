# LING 696G Final Project

For my final project, I wanted to build a text-to-speech (TTS) system. Because I used VITS for the mid-course excercise and had heard about lower performance and longer training times for other models like Tacotron, I decided to look for a relatively performant alternative and settled on the [DelightfulTTS](https://arxiv.org/abs/2207.04646) model, which has has an [in-development branch](https://github.com/coqui-ai/TTS/tree/delightful-tts) in the Coqui TTS package. I trained the model on the Dutch language Common Voice dataset so that I could get an idea of the models' performance compared to my previous attempt.

## Model and main variations
Like VITS, DelightfulTTS is an end-to-end text-to-speech model. The main distinguishing feature of DelightfulTTS is its focus on prosody and improving the performance of speech representation generation (in this case, mel-spectrograms) by generating down-sampled predictions which are then up-sampled by the vocoder. Because this approach creates a generalizable acoustic model and vocoder, explicit labels for speakers and languages can be supplied and the model can learn representations for individual speakers or specific languages while training on a large and varied dataset.

Since DelightfulTTS can leverage single-speaker information while training on a multi-speaker dataset, I decided to train two models on the full Dutch Common Voice dataset for comparison: one treating all speakers as a single speaker, and one that tracked speaker IDs and learned individual speaker embeddings. With my earlier VITS model, I managed to generate speech that was fairly convincing in tone and vocal quality by training on a single-speaker dataset; however, because of the limited amount of data, utterances were often indistinct and inference quickly degenerated when faced with situations like out-of-sample vocabulary or longer text inputs. A larger dataset could help resolve these issues, but without a way to single out speakers the result would likely sound inhuman. With an expanded dataset combined with DelightfulTTS's speaker embedding capacities, I hoped that I could have a model that could produce robust Dutch speech while also generating audio that sounded as if it came from a single person rather than a cacophony of speakers at once.

To get DelightfulTTS working, I needed to set up a new image that had the appropriate source code. I ran into errors in the branch that prevented inference from successfully producing audio, so I forked it and made a small change to get inference working—my image uses this branch, and can be set up using this this Singularity `.def` file (which I called `modified.def`):
```Singularity
Bootstrap: docker
From: python:3.9

%post 
    pip install https://github.com/mostol/TTS/archive/delightful-tts.zip
    pip install -f 'https://synesthesiam.github.io/prebuilt-apps/' gruut[nl]
```
To create the image (which I named `coqui-tts.sif`), you can run:
```shell
singularity coqui-tts.sif modified.def
```

With the Singularity image created, the models can be trained with:
```shell
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec --nv coqui-tts.sif python3 <scriptname.py>
```

## Parameters and additional variations
My training script was based of of an [existing DelightfulTTS recipe](https://github.com/coqui-ai/TTS/blob/delightful-tts/recipes/ljspeech/delightful_tts/train_delightful_tts.py) for the `ljspeech` dataset, with some modifications. Since my primary point of comparison was intended to be single- vs. multi-speaker setups, I tried to keep other parameters consistent between the two models and to select optimal configurations. Rather than using a character-based configuration, I used phonemes by taking advantage of `gruut`'s Dutch phoneme selection. This required installing an extension to the `gruut` package. My assumption was that using phonemes would allow the model to learn better speech generation more quickly by providing it a more unified speech representation—if the model uses phonemes, it doesn't have to learn to navigate the additinal variability that stems from trying to map orthography to pronunciation. For the dataset, I used the `validated.tsv` Common Voice data directly and leveraged Coqui's built-in `common_voice` dataset formatter for convenience and consistency. Finally, to ensure *ample* opportunity to produce the best model possible, I configured training to run for 100 epochs. Larger datsets often need fewer epochs—and in fact, these models converged and hit their maximum validation set performance well before they reached 100 epochs.

The changes I initially made to the existing recipe script are outlined in [original_training_script.diff](original_training_script.diff), while the actual scripts used for each model are available in their corresponding directories in the repo. In general, I set the dataset directory and formatter, reduced the batch size from the default to solve memory errors, altered audio preprocessing cache locations, and reduced the learning rate.

These changes reflect the setup needed for the "single-speaker" model—the multi-speaker model required some additional tweaks, which I based on a [recipe for the VTCK dataset](https://github.com/coqui-ai/TTS/blob/delightful-tts/recipes/vctk/delightful_tts/train_delightful_tts.py):
```diff
# ... in the imports...
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, DelightfulTTS, VocoderConfig
+ from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor
# ...

# ... before instantiating the trainer...
+ speaker_manager = SpeakerManager()
+ speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
+ config.model_args.num_speakers = speaker_manager.num_speakers

- model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=None)
+ model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=speaker_manager)

# trainer = Trainer(...)
```
With these settings, I was able to begin training runs without any immediate issues or errors. However, I ran into a few obstacles that led me to experiment with new model variations in hopes of achieving better performance.

### Batch size impact
One persistent problem that soon cropped up was an out-of-memory error that appeared after a few epochs. This issue cut training short, so the models were likely not attaining the performance they could be if the error were overcome. The out-of-memory error was a CUDA error, which meant it had to do with the amount of data in the GPU at one time, but it also only appeared at the end of an epoch (but before the evaluation steps began). Given the nature and timing of the error, I concluded that, while the GPU could handle size 16 batches, these were at the very limit of its memory and attempting to evaluate on additional size-16 batches pushed it over its capacities—so training would continue more smoothly if I lowered the batch size. I updated my configurations to use a batch size of 8, and was able to train much more successfully:
```diff
delightful_tts_config = DelightfulTTSConfig(
    # ...
-    batch_size=16, 
-    eval_batch_size=16,
+    batch_size=8, 
+    eval_batch_size=8,
    # ...
)
```
For comparison's sake, I've included some [sample outputs](./batchsize_16/) from a model that failed training because of this error.

### Learning rate impact
With the above settings applied, the models were ready to fully. While monitoring their progress, however, I noticed a consistent issue: after a few epochs, the models' loss would reach `nan` and either fail to improve for the remainder of training or cause session-ending errors. After further investigation, I noticed that, while I had set the learning rate low enough to avoid instant failure, the learning rate seemed to stay fixed as the model trained. With a fixed learning rate, the models were unable to improve their loss past a certain point because they would "overshoot" the weight updates needed to reduce their loss, and this continual overcorrection would eventually accumulate and spiral out of control.

Because of Coqui's default settings, this problem may not be seen in all cases. Coqui sets `scheduler_after_epoch` to `True` by default—this tells the trainer to update the learning rate *after each epoch*. An epoch is a single iteration over the entire dataset, so with smaller datasets, the learning rate will generally be adjusted before it becomes an issue. But when training on the entire Dutch Common Voice dataset (and with this particular combination of model architecture and training hardware) the distance between learning rate updates becomes long enough that the weight update step has the possibility of adjusting the weights to irrecoverably large or small values. Fortunately, this issue can be addressed by simply adjusting `scheduler_after_epoch` to `False`:
```diff
delightful_tts_config = DelightfulTTSConfig(
    # ...
    lr_gen=4e-4,
    lr=4e-4,
    lr_disc=4e-4,
+    scheduler_after_epoch=False,
)
```
This will allow the trainer to continually update the learning rate instead of waiting until the end of each epoch. Because this seemed to make a significant impact on model performance, I decided to include an ["epoch-scheduled" variation](./static_lr/) of the model among my comparison of options to see how it performs against the versions which have continual learning rate adjustments.

## Results
With so much effort spent on troubleshooting the models to get them functioning and training smoothly, I was excited to examine the end results comparing the single- vs. multi-speaker systems. Unfortunately, the experimental DelightfulTTS implementation was missing some components enabling the comparison. While the multi-speaker model was able to identify speakers, the speaker embeddings were either incorrectly trained or incorrectly stored, and I was unable to produce an output based on a specific speaker's embedding. I attempted to compensate by fine-tuning on a single speaker instead, but missing components in the model's configuration class mean that I was also unable to manage training from an existing DelightfulTTS model. This meant that the key differentiator between my two main variations—individualized vs. collective speech—was unable to be evaluated.

Despite this limitation, the two variants *were* trained with slightly different settings, one with a `SpeakerManager` and one without. This could, in theory, impact the models' performance—it is possible that tracking which audio came from which speaker could impact the underlying acoustic model and perhaps improve generalizability. However, a glance at the losses of the two models suggested that there was not a significant difference, and an auditory comparison confirms this: the results of the two variants on a few sample phrases are practically indistinguishable. (However, both models performed much better than the "oversized batch" and "static learning rate" variants, because they avoided training pitfalls that were causing severe performance issues.)

### Other comparisons
While model issues prevented these variants from being thoroughly compared, some simple comparisons can be drawn between this DelightfulTTS model trained on the full Dutch Common Voice dataset and my earlier VITS model trained on a single speaker. Notably, while the DelightfulTTS model's results have a "robotic" tone to them caused by combining all speakers together, the speech produced is indeed more complete and robust—all words are clearly and completely spoken, and they are also uttered with relatively natural prosody, despite the uncanny vocalization.
