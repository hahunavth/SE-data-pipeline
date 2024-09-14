import os

import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, read_audio
from tqdm import tqdm

model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
(get_speech_timestamps, _, read_audio, _, _) = utils


def vad_split(
    wav_path, output_dir=None, sampling_rate=48000, save=True, use_tqdm=False
):
    try:
        video_id = os.path.basename(wav_path).replace(".wav", "")

        wav_16k = read_audio(wav_path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav_16k,
            model,
            sampling_rate=16000,
            min_speech_duration_ms=500,
            max_speech_duration_s=15,
            min_silence_duration_ms=150,
        )

        wav_ori = read_audio(wav_path, sampling_rate=sampling_rate)
        scale = sampling_rate / 16000

        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        segments = []
        segments_meta = []

        prog = tqdm(speech_timestamps) if use_tqdm else speech_timestamps

        for i, ts in enumerate(prog):
            start = int(ts["start"] * scale)
            end = int(ts["end"] * scale)

            segment = wav_ori[start:end]

            if save and output_dir:
                seg_path = os.path.join(
                    output_dir, f"{video_id}_{str(i).zfill(8)}.wav"
                )
                sf.write(seg_path, segment, sampling_rate)
                segments.append(seg_path)
                segments_meta.append(ts)

        return segments, segments_meta
    except Exception as e:
        print(e)


if __name__ == "__main__":
    import sys

    wav_path = sys.argv[1]
    output_dir = sys.argv[2]
    fpaths = vad_split(wav_path, output_dir)
    print("======")
    for f in fpaths:
        print(f)
