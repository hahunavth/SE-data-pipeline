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
import logging.handlers
import math
import librosa
import torch
from tqdm import tqdm
import pandas as pd
from huggingface_hub import upload_folder, repo_exists, create_repo
from multiprocessing import Pool, get_context, Queue
import multiprocessing
import numpy as np

from audio_ac import classify_audio_batch
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import download_audio, get_youtube_playlist_ids

# Global variables for directories
download_dir = "tmp/downloaded"
segments_dir = "tmp/segments"

# Function to setup the logger in the main process (log listener)
def setup_logging(log_queue):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler
    f_handler = logging.FileHandler('tmp/pipeline.log')
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    # Stream handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.WARNING)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    # Queue handler for multiprocessing logging
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

# Function to configure the logger in each subprocess
def configure_subprocess_logger(log_queue):
    queue_listener = logging.handlers.QueueListener(log_queue, logging.getLogger())
    queue_listener.start()

# Function to process each channel (no change here)
def process_channel(row, min_snr, min_ac_speech_prob, log_queue):
    configure_subprocess_logger(log_queue)

    logger = logging.getLogger(__name__)
    all_channel_meta = {**row}
    all_channel_meta["videos"] = {}
    selected_channel_meta = {**row}
    selected_channel_meta["videos"] = {}

    channel_id = row["id"]
    channel_url = row["url"]
    logger.info(f"Processing channel {channel_id} ({channel_url})")

    try:
        video_ids = get_youtube_playlist_ids(channel_url)
        if len(video_ids) < 3:
            logger.warning(f"Channel {channel_id} has less than 3 videos")
            with open("tmp/skipped_channels.txt", "a") as f:
                f.write(f"{channel_id}\n")
            return all_channel_meta, selected_channel_meta

        audio_paths = [download_audio(f"https://www.youtube.com/watch?v={video_id}", download_dir) for video_id in video_ids[:3]]

        segments_path = []
        segments_meta_vad = []
        segments_video_id = []
        for audio_path, video_id in zip(audio_paths, video_ids):
            _segments_path, _segments_meta = vad_split(audio_path, output_dir=segments_dir)
            segments_path.extend(_segments_path)
            segments_meta_vad.extend(_segments_meta)
            segments_video_id.extend([video_id] * len(_segments_path))

        segments_snr = [estimate_snr(librosa.load(f)[0]) for f in segments_path]
        acss = classify_audio_batch(segments_path)
        torch.cuda.empty_cache()
        speech_probs = [item["score"] for ac in acss for item in ac if item["label"] == "Speech"]

        for i, (f, snr, speech_prob, acs, video_id) in enumerate(zip(segments_path, segments_snr, speech_probs, acss, segments_video_id)):
            is_selected = snr >= min_snr and speech_prob >= min_ac_speech_prob
            embed_url = f"https://www.youtube.com/embed/{video_id}?start={math.floor(segments_meta_vad[i]['start'] / 48000)}&end={math.ceil(segments_meta_vad[i]['end'] / 48000)}"
            all_channel_meta["videos"].setdefault(video_id, []).append({
                "idx": os.path.basename(f).replace(".wav", ""),
                "url": embed_url,
                "selected": is_selected,
                "vad": segments_meta_vad[i],
                "snr": snr,
                "ac": acs,
            })

            if is_selected:
                selected_channel_meta["videos"].setdefault(video_id, []).append({
                    "idx": os.path.basename(f).replace(".wav", ""),
                    "url": embed_url,
                    "start": segments_meta_vad[i]["start"],
                    "end": segments_meta_vad[i]["end"],
                })
            else:
                os.remove(f)

        for f in audio_paths:
            os.remove(f)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    return all_channel_meta, selected_channel_meta

# Main function
def main(df_or_path="tmp/yt_channels.csv", min_snr=15, min_ac_speech_prob=0.9, repo_id=None, verbose=False):
    log_queue = Queue()
    setup_logging(log_queue)

    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    all_segments_meta = {}
    selected_channels_meta = {}

    with get_context('spawn').Pool() as pool:
        results = pool.starmap(process_channel, [(row, min_snr, min_ac_speech_prob, log_queue) for _, row in df.iterrows()])

    for all_channel_meta, selected_channel_meta in results:
        channel_id = all_channel_meta["id"]
        all_segments_meta[channel_id] = all_channel_meta
        selected_channels_meta[channel_id] = selected_channel_meta

    # Save results
    with open("tmp/metadata_all.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(all_segments_meta, indent=4, ensure_ascii=False, default=convert_numpy))
    with open("tmp/metadata_selected.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(selected_channels_meta, indent=4, ensure_ascii=False, default=convert_numpy))

    # Upload to huggingface
    repo_type = "dataset"
    if not repo_exists(repo_id, repo_type=repo_type):
        create_repo(repo_id, repo_type=repo_type, private=True)
    upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path="tmp", path_in_repo="train")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
