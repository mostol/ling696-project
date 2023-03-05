import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.delightful_tts_config import DelightfulTtsAudioConfig, DelightfulTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, DelightfulTTS, VocoderConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor

# from TTS.tts.configs.shared_configs import CharactersConfig # <- Added

data_path = os.path.join("/home/u26/jmostoller/ondemand/ling696/cv-corpus-12.0-2022-12-07/nl")
output_path = os.path.join("/xdisk/hahnpowell/jmostoller/")

dataset_config = BaseDatasetConfig(
    formatter="common_voice", 
    meta_file_train="/home/u26/jmostoller/ondemand/ling696/cv-corpus-12.0-2022-12-07/nl/validated.tsv",
    path=data_path
)

# Left alone, but maybe worth experimenting with because of the audio source?
audio_config = DelightfulTtsAudioConfig()
model_args = DelightfulTtsArgs() # Also look at these...?


vocoder_config = VocoderConfig()

##here are the Dutch characters
#characters = CharactersConfig(
#    characters="AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZzÁÉÍÓÚáéíóúÄËÏÖÜäëïöü",
#    punctuations="?¬\";,- !.':",
#    characters_class=None, # Modified
#    pad="<PAD>",
#)

# Mess with these if it doesn't work at first.
delightful_tts_config = DelightfulTTSConfig(
    run_name="delightful_tts_nl_speakers",
    run_description="Train on Dutch from CommonVoice, using speakers",
    model_args=model_args,
    audio=audio_config,
    vocoder=vocoder_config,
    batch_size=8,
    eval_batch_size=8,
    num_loader_workers=1, # Does this depend on CPUs or something?
    num_eval_loader_workers=1,
    precompute_num_workers=1,
    batch_group_size=0,
    compute_input_seq_cache=True,
    compute_f0=True, # Maybe just skip this at first?
    # f0_cache_path=None,
    f0_cache_path=os.path.join(data_path, "f0_cache/"),
    run_eval=True,
    # test_delay_epochs=-1,
    epochs=100, # Default was 1000, but I've bumped it down.
    text_cleaner="phoneme_cleaners", # Is there a better cleaner option?
    # use_phonemes=False,
    # characters=characters,
    use_phonemes=True,
    phoneme_language="nl", # Trying out phonemes—this might be one aspect to vary.
    phoneme_cache_path=os.path.join(data_path, "phoneme_cache"),
    print_step=100, # Default was 50.
    print_eval=False, # Was False
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    start_by_longest=False,
    eval_split_size=0.1,
    binary_align_loss_alpha=0.0,
    use_attn_priors=False,
    lr_gen=4e-4,
    lr=4e-4,
    lr_disc=4e-4,
    scheduler_after_epoch=False,
    #min_text_len=0,
    #max_text_len=500,
    #min_audio_len=0,
    #max_audio_len=500000,
)

tokenizer, config = TTSTokenizer.init_from_config(delightful_tts_config)

ap = AudioProcessor.init_from_config(config)


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Speaker Manager
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=speaker_manager)

trainer = Trainer(
    TrainerArgs(), 
    config, 
    output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

trainer.fit()