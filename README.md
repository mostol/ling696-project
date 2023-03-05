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

## Model and main variations
Like VITS, DelightfulTTS is an end-to-end text-to-speech model. The main distinguishing feature of DelightfulTTS is its focus on prosody and improving the performance of speech representation generation (in this case, mel-spectrograms) by generating down-sampled predictions which are then up-sampled by the vocoder. Because this approach creates a generalizable acoustic model and vocoder, explicit labels for speakers and languages can be supplied and the model can learn representations for individual speakers or specific languages while training on a large and varied dataset.

Since DelightfulTTS can leverage single-speaker information while training on a multi-speaker dataset, I decided to train two models on the full Dutch Common Voice dataset for comparison: one treating all speakers as a single speaker, and one that tracked speaker IDs and learned individual speaker embeddings. With my earlier VITS model, I managed to generate speech that was fairly convincing in tone and vocal quality by training on a single-speaker dataset; however, because of the limited amount of data, utterances were often indistinct and inference quickly degenerated when faces with situations like out-of-sample vocabulary or longer text inputs. A larger dataset could help resolve these issues, but without a way to single out speakers the result would likely be inhuman-sounding. With an expanded dataset combined with DelightfulTTS's speaker embedding capacities, I hoped that I could have a model that could produce robust Dutch speech while also generating audio that sounded as if it came from a single person rather than a cacophony of speakers at once.

To get DelightfulTTS working, I needed to set up a new image that had the appropriate source code. I ran into errors in the branch that prevented inference from successfully producing audio, so I forked it and made a small change to get inference working—my image uses this branch, and can be set up using this this Singularity `.def` file (which I called `modified.def`):
```Singularity
Bootstrap: docker
From: python:3.9

%post 
    pip install https://github.com/mostol/TTS/archive/delightful-tts.zip
    pip install -f 'https://synesthesiam.github.io/prebuilt-apps/' gruut[nl]
```

## Parameters and additional variations
My training script was based of of an [existing DelightfulTTS recipe](https://github.com/coqui-ai/TTS/blob/delightful-tts/recipes/ljspeech/delightful_tts/train_delightful_tts.py) for the `ljspeech` dataset, with some modifications. Since my primary point of comparison was intended to be single- vs. multi-speaker setups, I tried to keep other parameters consistent between the two models and to select optimal configurations:
* Rather than using a character-based configuration, I used phonemes by taking advantage of `gruut`'s Dutch phoneme selection. This required installing an extension to the `gruut` package. My assumption was that using phonemes would allow the model to learn better speech generation more quickly by providing it a more unified speech representation—if the model uses phonemes, it doesn't have to learn to navigate the additinal variability that stems from trying to map orthography to pronunciation.
* For the dataset, I used the `validated.tsv` Common Voice data directly and leveraged Coqui's built-in `common_voice` dataset formatter for convenience and consistency.
* To ensure *ample* opportunity to produce the best model possible, I configured training to run for 100 epochs. Larger datsets often need fewer epochs—and in fact, these models converged and hit their maximum validation set performance well before they reached 100 epochs.

The changes I initially made to the existing recipe script are outlined below (the actual scripts used for each model are available in their corresponding directories in the repo):

```diff
import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.delightful_tts_config import DelightfulTtsAudioConfig, DelightfulTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, DelightfulTTS, VocoderConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor

- data_path = ""
+ # This points to the "root" directory of the Common Voice data—the one that contains `validated.tsv`, `clips`, etc.
+ data_path = os.path.join("/home/u26/jmostoller/ondemand/ling696/cv-corpus-12.0-2022-12-07/nl") 
- output_path = os.path.dirname(os.path.abspath(__file__))
+ output_path = os.path.join("/xdisk/hahnpowell/jmostoller/") # Each model is output to its own directory on xdisk

dataset_config = BaseDatasetConfig(
-    dataset_name="ljspeech", 
-    formatter="ljspeech", 
+    formatter="common_voice", # Use the Common Voice auto-formatting to read the meta file/dataset as-is
-    meta_file_train="metadata.csv", 
+    # Change the training file to point to the `valitaded.tsv` file associated with the dataset:
+    meta_file_train="/home/u26/jmostoller/ondemand/ling696/cv-corpus-12.0-2022-12-07/nl/validated.tsv",
    path=data_path
)

audio_config = DelightfulTtsAudioConfig()
model_args = DelightfulTtsArgs()

vocoder_config = VocoderConfig()

delightful_tts_config = DelightfulTTSConfig(
+    run_name="delightful_tts_nl", # Custom run name set :)
+    run_description="Train on Dutch from CommonVoice",
    model_args=model_args,
    audio=audio_config,
    vocoder=vocoder_config,
-    batch_size=32,
+    # 32-sample batches were too big to even start training on, 
+    batch_size=16, 
+    eval_batch_size=16,
-    num_loader_workers=10,
-    num_eval_loader_workers=10,
-    precompute_num_workers=10,
+    # Coqui warned that my system was only able to use 1 worker, so I lowered all these values.
+    num_loader_workers=1,
+    num_eval_loader_workers=1,
+    precompute_num_workers=1,
-    batch_group_size=2,
+    batch_group_size=0, # Batch gruping seemed to be causing memory issues, so I opted out of it.
    compute_input_seq_cache=True,
    compute_f0=True,
-    f0_cache_path=os.path.join(output_path, "f0_cache"),
+    # Storing audio-related caches with the dataset helps reusability
+    f0_cache_path=os.path.join(data_path, "f0_cache"), 
    run_eval=True,
    test_delay_epochs=-1,
-    epochs=1000,
+    epochs=100, # Reduced the number of epochs; 100 is plenty!
-    text_cleaner="english_cleaners",
+    text_cleaner="phoneme_cleaners", # No built-in Dutch cleaner, so I used a more generic "phoneme" one
    use_phonemes=True,
-    phoneme_language="en-us",
+    phoneme_language="nl", # Use Dutch phonemes! 🇳🇱
-    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
+    phoneme_cache_path=os.path.join(data_path, "phoneme_cache"), # Cache phonemes with the dataset
-    print_step=50,
+    print_step=100, # Increase the print steps (there are a lot of steps!)
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    start_by_longest=False,
    eval_split_size=0.1,
    binary_align_loss_alpha=0.0,
    use_attn_priors=False,
-    lr_gen=4e-1,
-    lr=4e-1,
-    lr_disc=4e-1,
+    # These learning rates were *incredibly* high and would instantly lead to errors unless lowered.
+    lr_gen=4e-4,
+    lr=4e-4,
+    lr_disc=4e-4,
-    max_text_len=130, # I wasn't sure about the max text length, so I left it unspecified.
)

tokenizer, config = TTSTokenizer.init_from_config(delightful_tts_config)

ap = AudioProcessor.init_from_config(config)


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

train_samples = train_samples
eval_samples = eval_samples

model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
```

These changes reflect the setup for the "single-speaker" model—the multi-speaker model required some additional tweaks, which I based on a [recipe for the VTCK dataset](https://github.com/coqui-ai/TTS/blob/delightful-tts/recipes/vctk/delightful_tts/train_delightful_tts.py):
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

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
```
With these settings, I was able to begin training runs without any immediate issues or errors. However, I ran into a few obstacles that led me to experiment with new model variations in hopes of achieveing better performance.

### Batch size impact
One persistent problem that soon cropped up was an out-of-memory error that appeared after a few epochs. This issue cut training short, so the models were likely not attaining the performance they could be if the error were overcome. The out-of-memory error was a CUDA error, which meant it had to do with the amount of data in the GPU at one time, but it also only appeared at the end of an epoch (but before the evaluation steps began). Given the nature and timing of the error, I concluded that, while the GPU could handle size 16 batches, these were at the very limit of its memory and attemptign to evaluate on additional size-16 batches pushed it over its capacities—so training would continue more smoothly if I lowered the batch size. I updated my configurations to use a batch size of 8, and was able to train much more successfully:
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
For comparison's sake, I've included some sample outputs from a model that failed training because of this error.

### Learning rate impact
With the above settings applied, the models were ready to fully. While monitoring their progress, however, I noticed a consistent issue: after a few epochs, the models' loss would reach `nan` and either fail to improve for the remainder of training or cause session-ending errors. After further investigation, I noticed that, while I had set the learning rate low enough to avoid instant failure, the learning rate seemed to stay fixed as the model trained. With a fixed learning rate, the models were unable to improve their loss past a certain point because they would "overshoot" the weight updates needed to reduce their loss, and this continual overcorrection would eventually accumulate and spiral out of control.

Because of Coqui's default settings, this problem may not be seen in all cases. Coqui sets `scheduler_after_epoch` to `True` by default—this tells the trainer to update the learning rate *after each epoch*. An epoch is a single iteration over the entire dataset, so with smaller datasets, the learning rate will generally be adjusted before it becomes an issue. But when training on the entire Dutch Common Voice dataset (and with this particular combination of model architecture and training hardware) the distance between learning rate updates becomes long enough that the weight update step has the possibility of adjusting the weights to irrecoverably large or small values. Fortunately, this issue can be addressed by simply adjusting `scheduler_after_epoch` to `False`:
```diff
delightful_tts_config = DelightfulTTSConfig(
    # ...
    # ...
    lr_gen=4e-4,
    lr=4e-4,
    lr_disc=4e-4,
+    scheduler_after_epoch=False,
)
```
this will allow the trainer to continually update the learning rate instead of waiting until the end of each epoch. Because this seemed to make a significant impact on model performance, I decided to include an "epoch-scheduled" variation of the model among my comparison of options to see how it performs against the versions which have continual learning rate adjustments.

## Results
With so much effort spent on troubleshooting the models to get them functioning and training smoothly, I was excited to examine the end results comparing the single- vs. multi-speaker systems. Unfortunately, the experimental DelightfulTTS implementation was missing some components enabling the comparison. While the multi-speaker model was able to identify speakers, the speaker embeddings were either incorrectly trained or incorrectly stored, and I was unable to produce an output based on a specific speaker's embedding. I attempted to compensate by fine-tuning on a single speaker instead, but missing components in the model's configuration class mean that I was also unable to manage training from an existing DelightfulTTS model. This meant that the key differentiator between my two main variations—individualized vs. collective speech—was unable to be evaluated. However, the two variants *were* trained with slightly different settings, one with a `SpeakerManager` and one without. A glance at the losses of the two models suggested that there was not a significant difference, and an auditory comparison confirms this: the results of the two variants on a few sample phrases are practically indistinguishable.
