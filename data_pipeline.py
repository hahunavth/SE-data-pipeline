# import json
# import os
# import logging
# import math

# import librosa
# import torch
# from tqdm import tqdm
# import pandas as pd
# from huggingface_hub import upload_folder, repo_exists, create_repo

# from audio_ac import classify_audio_batch
# from audio_snr import estimate_snr
# from audio_vad import vad_split
# from yt_download import (
#     download_and_cut_n_audio,
#     get_youtube_playlist_ids,
#     check_audio_quality_48k,
#     download_audio,
# )

# # Create a custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # Create handlers
# c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler('tmp/pipeline.log')
# c_handler.setLevel(logging.WARNING)   # Console handler level is set to INFO
# f_handler.setLevel(logging.ERROR)  # File handler level is set to ERROR

# # Create formatters and add it to handlers
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)
# f_handler.setFormatter(f_format)

# # Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)


# def main(
#     df_or_path="tmp/yt_channels.csv",
#     # n video select startegy
#     n_per_10ksubs_plus=1,
#     n_per_20ksubs_plus=2,
#     n_per_50ksubs_plus=5,
#     n_per_100ksubs_plus=10,
#     # vad
#     min_duration=None,
#     max_duration=None,
#     # threshold
#     min_snr=15,
#     min_ac_speech_prob=0.9,
#     # hf
#     split="train",
#     repo_id=None,
#     # log
#     verbose=False,
# ):
#     if verbose:
#         logger.setLevel(logging.DEBUG)
#         c_handler.setLevel(logging.DEBUG)
#         f_handler.setLevel(logging.DEBUG)

#     if isinstance(df_or_path, str):
#         df = pd.read_csv(df_or_path)
#     else:
#         df = df_or_path

#     download_dir = "tmp/downloaded"
#     segments_dir = "tmp/segments"

#     all_segments_meta = {} # huggingface dataset meta (sample level)
#     selected_channels_meta = {} # external dataset meta (tree: channel -> video -> segment)

#     for i, row in tqdm(df.iterrows(), total=len(df)):
#         all_channel_meta = {**row}
#         all_channel_meta["videos"] = {}
#         selected_channel_meta = {**row}
#         selected_channel_meta["videos"] = {}

#         channel_id = row["id"]
#         channel_custom_id = row["custom_url"] # @alias
#         channel_url = row["url"]
#         logger.info(f"Processing channel {channel_id} ({channel_url})")
#         # print(get_youtube_playlist_ids(url))

#         logger.debug("Getting video ids")
#         video_ids = get_youtube_playlist_ids(channel_url)
#         if len(video_ids) < 3:
#             logger.warning(f"Channel {channel_id} has less than 3 videos")
#             # skip
#             # TODO: log
#             with open("tmp/skipped_channels.txt", "a") as f:
#                 f.write(f"{channel_id}\n")
#             continue

#         logger.debug("Downloading audio")
#         audio_paths = []
#         for video_id in video_ids[:3]: # download 3 audios
#             try:
#                 audio_path = download_audio(f"https://www.youtube.com/watch?v={video_id}", download_dir)
#                 audio_paths.append(audio_path)
#             except Exception as e:
#                 logger.error(f"An error occurred: {e}")

#         # vad
#         logger.debug("VAD")
#         segments_path = []
#         segments_meta_vad = []
#         segments_video_id = []
#         for audio_path, video_id in zip(audio_paths, video_ids):
#             try:
#                 _segments_path, _segments_meta = vad_split(
#                     audio_path,
#                     output_dir=segments_dir,
#                     # min_duration=min_duration,
#                     # max_duration=max_duration,
#                 )
#                 # add video url
#                 segments_path.extend(_segments_path)
#                 segments_meta_vad.extend(_segments_meta)
#                 segments_video_id.extend([video_id] * len(_segments_path))
#             except Exception as e:
#                 logger.error(f"An error occurred: {e}")

#         # snr
#         logger.debug("Estimating SNR")
#         segments_snr = [estimate_snr(librosa.load(f)[0]) for f in segments_path]
#         logger.debug(f"SNR: {segments_snr}")

#         # ac
#         logger.debug("Classifying audio")
#         acss = classify_audio_batch(segments_path)
#         torch.cuda.empty_cache()
#         speech_probs = []
#         for ac in acss:
#             for item in ac:
#                 if item["label"] == "Speech":
#                     speech_probs.append(item["score"])
#         logger.debug(f"AC: {acss}")

#         # refine
#         # channel_refined_ds_segments_meta = []
#         for i, (f, snr, speech_prob, acs, video_id) in enumerate(zip(segments_path, segments_snr, speech_probs, acss, segments_video_id)):
#             is_selected = snr >= min_snr and speech_prob >= min_ac_speech_prob
#             embed_url = f"https://www.youtube.com/embed/{video_id}?start={math.floor(segments_meta_vad[i]['start'] / 48000)}&end={math.ceil(segments_meta_vad[i]['end'] / 48000)}"
#             if video_id not in selected_channel_meta["videos"]:
#                 all_channel_meta["videos"][video_id] = []
#                 selected_channel_meta["videos"][video_id] = []

#             all_channel_meta["videos"][video_id].append(
#                 {
#                     "idx": os.path.basename(f).replace(".wav", ""),
#                     "url": embed_url,
#                     "selected": is_selected,
#                     "vad": segments_meta_vad[i],
#                     "snr": snr,
#                     "ac": acs,
#                 }
#             )

#             if is_selected:
#                 selected_channel_meta["videos"][video_id].append(
#                     {
#                         "idx": os.path.basename(f).replace(".wav", ""),
#                         "url": embed_url,
#                         "start": segments_meta_vad[i]["start"],
#                         "end": segments_meta_vad[i]["end"],
#                     }
#                 )

#                 # channel_refined_ds_segments_meta.append(
#                 #     {
#                 #         "idx": os.path.basename(f).replace(".wav", ""),
#                 #         "channel_id": channel_id,
#                 #         "channel_custom_id": channel_custom_id,
#                 #         "video_id": video_id,
#                 #         # "vad": segments_meta_vad[i],
#                 #         "start": segments_meta_vad[i]["start"],
#                 #         "end": segments_meta_vad[i]["end"],
#                 #         # snr
#                 #         "snr": snr,
#                 #         # "ac": acs,
#                 #         "ac_speech_prob": speech_prob,
#                 #         # TODO: asr
#                 #     }
#                 # )
#             else:
#                 os.remove(f)

#         # add to global list
#         # ds_segments_meta.extend(channel_refined_ds_segments_meta)
#         all_segments_meta[channel_id] = all_channel_meta
#         selected_channels_meta[channel_id] = selected_channel_meta
#         # clean
#         for f in audio_paths:
#             os.remove(f)

#     if os.path.exists(download_dir):
#         os.rmdir(download_dir)

#     # save ext_channels_meta json
#     with open("tmp/metadata_all.json", "w", encoding='utf-8') as f:
#         f.write(json.dumps(all_segments_meta, indent=4, ensure_ascii=False))
#     with open("tmp/metadata_selected.json", "w", encoding='utf-8') as f:
#         f.write(json.dumps(selected_channels_meta, indent=4, ensure_ascii=False))
#     # upload to huggingface
#     repo_type = "dataset"
#     if not repo_exists(repo_id, repo_type=repo_type):
#         create_repo(repo_id, repo_type=repo_type, private=True)
#     upload_folder(
#         repo_id=repo_id,
#         repo_type=repo_type,
#         folder_path="tmp",
#         path_in_repo=split,
#     )
#     os.rmdir("tmp")

# if __name__ == "__main__":
#     from fire import Fire
#     Fire(main)


# ========================================================================================================
# TRY MULTIPROCESSING

import json
import os
import logging
import math
import librosa
import torch
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, Manager
from huggingface_hub import upload_folder, repo_exists, create_repo

from audio_ac import classify_audio_batch
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import download_audio, get_youtube_playlist_ids

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('tmp/pipeline.log')
c_handler.setLevel(logging.WARNING)   # Console handler level is set to INFO
f_handler.setLevel(logging.ERROR)  # File handler level is set to ERROR

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Download audio function for multiprocessing
def download_video_audio(video_id, download_dir):
    try:
        audio_path = download_audio(f"https://www.youtube.com/watch?v={video_id}", download_dir)
        return audio_path
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        return None

# VAD processing function for multiprocessing
def process_vad(audio_info):
    audio_path, video_id, segments_dir = audio_info
    try:
        _segments_path, _segments_meta = vad_split(
            audio_path,
            output_dir=segments_dir
        )
        return _segments_path, _segments_meta, video_id
    except Exception as e:
        logger.error(f"An error occurred during VAD: {e}")
        return [], [], video_id

# SNR estimation function for multiprocessing
def estimate_snr_for_audio(audio_path):
    try:
        return estimate_snr(librosa.load(audio_path)[0])
    except Exception as e:
        logger.error(f"An error occurred during SNR estimation: {e}")
        return None

def main(
    df_or_path="tmp/yt_channels.csv",
    n_per_10ksubs_plus=1,
    n_per_20ksubs_plus=2,
    n_per_50ksubs_plus=5,
    n_per_100ksubs_plus=10,
    min_duration=None,
    max_duration=None,
    min_snr=15,
    min_ac_speech_prob=0.9,
    split="train",
    repo_id=None,
    verbose=False,
    num_workers=4,  # Number of parallel workers
):
    if verbose:
        logger.setLevel(logging.DEBUG)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    download_dir = "tmp/downloaded"
    segments_dir = "tmp/segments"

    all_segments_meta = {}  # huggingface dataset meta (sample level)
    selected_channels_meta = {}  # external dataset meta (tree: channel -> video -> segment)

    manager = Manager()
    all_segments_meta_mp = manager.dict()  # multiprocessing-safe dictionary
    selected_channels_meta_mp = manager.dict()  # multiprocessing-safe dictionary

    def process_channel(row):
        channel_id = row["id"]
        channel_custom_id = row["custom_url"]
        channel_url = row["url"]
        logger.info(f"Processing channel {channel_id} ({channel_url})")

        all_channel_meta = {**row, "videos": {}}
        selected_channel_meta = {**row, "videos": {}}

        logger.debug("Getting video ids")
        video_ids = get_youtube_playlist_ids(channel_url)
        if len(video_ids) < 3:
            logger.warning(f"Channel {channel_id} has less than 3 videos")
            with open("tmp/skipped_channels.txt", "a") as f:
                f.write(f"{channel_id}\n")
            return

        logger.debug("Downloading audio")
        with Pool(num_workers) as pool:
            audio_paths = pool.map(lambda video_id: download_video_audio(video_id, download_dir), video_ids[:3])

        audio_paths = [p for p in audio_paths if p]  # Remove failed downloads

        if not audio_paths:
            return

        logger.debug("VAD processing")
        vad_info = [(path, video_id, segments_dir) for path, video_id in zip(audio_paths, video_ids)]
        with Pool(num_workers) as pool:
            vad_results = pool.map(process_vad, vad_info)

        segments_path, segments_meta_vad, segments_video_id = [], [], []
        for _segments_path, _segments_meta, video_id in vad_results:
            segments_path.extend(_segments_path)
            segments_meta_vad.extend(_segments_meta)
            segments_video_id.extend([video_id] * len(_segments_path))

        logger.debug("Estimating SNR")
        with Pool(num_workers) as pool:
            segments_snr = pool.map(estimate_snr_for_audio, segments_path)

        logger.debug("Classifying audio")
        acss = classify_audio_batch(segments_path)
        torch.cuda.empty_cache()
        speech_probs = []
        for ac in acss:
            for item in ac:
                if item["label"] == "Speech":
                    speech_probs.append(item["score"])

        for i, (f, snr, speech_prob, acs, video_id) in enumerate(zip(segments_path, segments_snr, speech_probs, acss, segments_video_id)):
            is_selected = snr >= min_snr and speech_prob >= min_ac_speech_prob
            embed_url = f"https://www.youtube.com/embed/{video_id}?start={math.floor(segments_meta_vad[i]['start'] / 48000)}&end={math.ceil(segments_meta_vad[i]['end'] / 48000)}"
            if video_id not in selected_channel_meta["videos"]:
                all_channel_meta["videos"][video_id] = []
                selected_channel_meta["videos"][video_id] = []

            all_channel_meta["videos"][video_id].append({
                "idx": os.path.basename(f).replace(".wav", ""),
                "url": embed_url,
                "selected": is_selected,
                "vad": segments_meta_vad[i],
                "snr": snr,
                "ac": acs,
            })

            if is_selected:
                selected_channel_meta["videos"][video_id].append({
                    "idx": os.path.basename(f).replace(".wav", ""),
                    "url": embed_url,
                    "start": segments_meta_vad[i]["start"],
                    "end": segments_meta_vad[i]["end"],
                })
            else:
                os.remove(f)

        all_segments_meta_mp[channel_id] = all_channel_meta
        selected_channels_meta_mp[channel_id] = selected_channel_meta

        for f in audio_paths:
            os.remove(f)

    # Process each channel with multiprocessing
    with Pool(num_workers) as pool:
        pool.map(process_channel, [row for _, row in df.iterrows()])

    # Convert multiprocessing-safe dicts to regular dicts for saving
    all_segments_meta = dict(all_segments_meta_mp)
    selected_channels_meta = dict(selected_channels_meta_mp)

    # Save metadata
    with open("tmp/metadata_all.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(all_segments_meta, indent=4, ensure_ascii=False))
    with open("tmp/metadata_selected.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(selected_channels_meta, indent=4, ensure_ascii=False))

    # Upload to Huggingface
    repo_type = "dataset"
    if not repo_exists(repo_id, repo_type=repo_type):
        create_repo(repo_id, repo_type=repo_type, private=True)
    upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path="tmp",
        path_in_repo=split,
    )
    os.rmdir("tmp")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
