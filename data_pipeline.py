import json
import os

import librosa
import torch
from tqdm import tqdm
import pandas as pd

from audio_ac import classify_audio_batch
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import (
    download_and_cut_n_audio,
    get_youtube_playlist_ids,
    check_audio_quality_48k,
    download_audio,
)


def main(
    df_or_path="tmp/yt_channels.csv",
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
    # hf
    split="train",
):
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    download_dir = "tmp/downloaded"
    segments_dir = "tmp/vad"

    ds_segments_meta = [] # huggingface dataset meta (sample level)
    ext_channels_meta = {} # external dataset meta (tree: channel -> video -> segment)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        channel_meta = {**row}
        channel_meta["videos"] = {}

        channel_id = row["id"]
        channel_custom_id = row["custom_url"] # @alias
        channel_url = row["url"]
        print(channel_url)
        # print(get_youtube_playlist_ids(url))

        video_ids = get_youtube_playlist_ids(channel_url)
        if len(video_ids) < 3:
            print(f"Channel {channel_id} has less than 3 videos")
            # skip
            # TODO: log
            with open("tmp/skipped_channels.txt", "a") as f:
                f.write(f"{channel_id}\n")
            continue

        audio_paths = []
        for video_id in video_ids[:3]: # download 3 audios
            try:
                audio_path = download_audio(f"https://www.youtube.com/watch?v={video_id}", download_dir)
                audio_paths.append(audio_path)
            except Exception as e:
                print(f"An error occurred: {e}")

        # vad
        segments_path = []
        segments_meta_vad = []
        segments_video_id = []
        for audio_path, video_id in zip(audio_paths, video_ids):
            try:
                _segments_path, _segments_meta = vad_split(
                    audio_path,
                    output_dir=segments_dir,
                    # min_duration=min_duration,
                    # max_duration=max_duration,
                )
                # add video url
                segments_path.extend(_segments_path)
                segments_meta_vad.extend(_segments_meta)
                segments_video_id.extend([video_id] * len(_segments_path))
            except Exception as e:
                print(f"An error occurred: {e}")

        # snr
        segments_snr = [estimate_snr(librosa.load(f)[0]) for f in segments_path]

        # ac
        acss = classify_audio_batch(segments_path)
        torch.cuda.empty_cache()
        speech_probs = []
        for ac in acss:
            for item in ac:
                if item["label"] == "Speech":
                    speech_probs.append(item["score"])

        # refine
        channel_refined_ds_segments_meta = []
        for i, (f, snr, speech_prob, acs, video_id) in enumerate(zip(segments_path, segments_snr, speech_probs, acss, segments_video_id)):
            is_selected = snr >= min_snr and speech_prob >= min_ac_speech_prob
            # add to channel meta
            if video_id not in channel_meta["videos"]:
                channel_meta["videos"][video_id] = []
            channel_meta["videos"][video_id].append(
                {
                    "idx": os.path.basename(f).replace(".wav", ""),
                    "vad": segments_meta_vad[i],
                    "snr": snr,
                    "ac": acs,
                }
            )
            
            if is_selected:
                channel_refined_ds_segments_meta.append(
                    {
                        "idx": os.path.basename(f).replace(".wav", ""),
                        "channel_id": channel_id,
                        "channel_custom_id": channel_custom_id,
                        "video_id": video_id,
                        # "vad": segments_meta_vad[i],
                        "start": segments_meta_vad[i]["start"],
                        "end": segments_meta_vad[i]["end"],
                        # snr
                        "snr": snr,
                        # "ac": acs,
                        "ac_speech_prob": speech_prob,
                        # TODO: asr
                    }
                )

        # add to global list
        ds_segments_meta.extend(channel_refined_ds_segments_meta)
        ext_channels_meta[channel_id] = channel_meta
        # clean
        for f in audio_paths:
            os.remove(f)

    # save ext_channels_meta json
    with open("tmp/ext_channels_meta.json", "w") as f:
        json.dump(ext_channels_meta, f)
    # upload to huggingface
    from datasets import Dataset
    ds = Dataset.from_dict(ds_segments_meta)
    ds.map(lambda x: {"audio": librosa.load(f"{segments_dir}/{x['idx']}.wav")[0]}) # ,remove_columns=["idx"]
    ds.save_to_disk("tmp/ds")
    ds.push_to_hub("hahunavth/se_yt_v1", use_temp_dir=True, split=split) 

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
