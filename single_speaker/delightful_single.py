import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.delightful_tts_config import DelightfulTtsAudioConfig, DelightfulTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, DelightfulTTS, VocoderConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor

# Set up the data and output paths. The data path needs to be the "root"
# folder of the Common Voice dataset (i.e. the folder that contains `clips` and `validation.tsv`).
data_path = os.path.join("/home/u26/jmostoller/ondemand/ling696/cv-corpus-12.0-2022-12-07/nl")
output_path = os.path.join("/xdisk/hahnpowell/jmostoller/") # I used xdisk for more space :)

dataset_config = BaseDatasetConfig(
    formatter="common_voice",  # The commonvoice formatter requires you to store your wav files in the `clips` folder.
    # If you leave everything in place and have your wavs in `clips`, you can use the Common Voice tsv files directly.
    meta_file_train="/home/u26/jmostoller/ondemand/ling696/cv-corpus-12.0-2022-12-07/nl/validated.tsv", # 
    path=data_path
)

# These default configs work just fine!
audio_config = DelightfulTtsAudioConfig()
model_args = DelightfulTtsArgs()

vocoder_config = VocoderConfig()

delightful_tts_config = DelightfulTTSConfig(
    run_name="delightful_tts_nl_speakers", # Set a run name
    run_description="Train on Dutch from CommonVoice, using speakers", # And a description
    model_args=model_args,
    audio=audio_config,
    vocoder=vocoder_config,
    batch_size=8, # Batch size set to 8 to avoid memory issues.
    eval_batch_size=8, # Ditto.
    num_loader_workers=1, # Coqui warned of "unused" workers, so I set these all to 1.
    num_eval_loader_workers=1,
    precompute_num_workers=1,
    batch_group_size=0, # Batch group size seemed to contribute to memory issues, so I turned it off.
    compute_input_seq_cache=True,
    compute_f0=True, # Computing f0 was done in the example recipe, so I kept it.
    # Caching the f0 files along with the training data makes it easy to reuse them later:
    f0_cache_path=os.path.join(data_path, "f0_cache/"), 
    run_eval=True,
    epochs=100, # Default was 1000, but I've bumped it down.
    text_cleaner="phoneme_cleaners", # There is no built-in `dutch_cleaners`, so I use the more generic `phoneme_cleaners`
    use_phonemes=True,
    phoneme_language="nl", # Requires gruut[nl] to be installed.
    phoneme_cache_path=os.path.join(data_path, "phoneme_cache"), # Like f0, it's handy to store these with the dataset.
    print_step=100, # Default was 50, which was a lot of printing.
    print_eval=False, # No need to print every single part of evaluation, jsut the results are fine.
    mixed_precision=True, # Mixed precision speeds things up!
    output_path=output_path,
    datasets=[dataset_config],
    start_by_longest=False, # This is a data sorting option that I found unecessary.
    eval_split_size=0.1,
    binary_align_loss_alpha=0.0,
    use_attn_priors=False,
    lr_gen=4e-4, # Lowered learning rates from default.
    lr=4e-4,
    lr_disc=4e-4,
    scheduler_after_epoch=False, # Critical for avoiding nan loss.
)

tokenizer, config = TTSTokenizer.init_from_config(delightful_tts_config)

ap = AudioProcessor.init_from_config(config)


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# No speaker manager here.

model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=none)

trainer = Trainer(
    TrainerArgs(), 
    config, 
    output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

trainer.fit()