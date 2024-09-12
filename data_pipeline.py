import json
import os

import librosa  # Ensure this import is here if you're using librosa for loading audio
import torch

from ac import classify_audio_batch
from snr import estimate_snr
from vad import vad_split
from youtube import download_and_cut_n_audio

def main(
    df,
    # n video select startegy
    n_per_10ksubs_plus=1,
    n_per_20ksubs_plus=2,
    n_per_50ksubs_plus=5,
    n_per_100ksubs_plus=10,
    # vad
    min_duration=None,
    max_duration=None,
    # threshold
    min_snr=15, 
    min_ac_speech_prob=0.9,
):
    
    pass