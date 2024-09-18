# %%writefile SE-data-pipeline/data_pipeline.py

import multiprocessing
from multiprocessing import Manager
import logging
import json
import os
import random
import torch
import librosa
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from huggingface_hub import upload_folder, repo_exists, create_repo
from audio_ac import ac_infer_batch, ac_get_speech_probs
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import yt_download_audio, yt_get_playlist_ids, yt_get_video_duration_sec
from hf import upload_folder_retry

def get_process_dirs(idx):
    tmp_dir = f"tmp/{idx}"
    download_dir = f"tmp/{idx}/downloaded"
    segments_dir = f"tmp/{idx}/segments"
    meta_list_dir = f"tmp/{idx}/meta_list" # store metadata for each channel in a file
    return tmp_dir, download_dir, segments_dir, meta_list_dir

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers for logging
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('tmp/pipeline.log')

# Set log levels
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.INFO)

# Create formatters and add to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)
# Function to handle logging in subprocesses
def log_listener(queue):
    while True:
        try:
            # Get the log record from the queue
            record = queue.get()
            if record is None:  # Sentinel to stop listener
                break
            # Ensure levelno is not None and has a valid value
            if record.levelno is None:
                record.levelno = logging.INFO  # Default to INFO if not set
            # Handle the log record
            logger.handle(record)
        except Exception as e:
            import sys
            import traceback
            print(f"Error in log listener: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def process_channel(row, min_snr, min_ac_speech_prob, log_queue, repo_id=None, split="test", channel_min_videos=5):
    tmp_dir, download_dir, segments_dir, meta_list_dir = get_process_dirs(row["id"])
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(meta_list_dir, exist_ok=True)

    all_channel_meta = {**row}
    all_channel_meta["videos"] = {}
    selected_channel_meta = {**row}
    selected_channel_meta["videos"] = {}

    channel_id = row["id"]
    channel_url = row["url"]
    channel_n_sub = row["n_subs"]
    channel_n_videos = row["n_videos"]
    channel_custom_url = row["custom_url"] if "custom_url" in row else None

    n_video_download = min(5, channel_n_sub // 10000 + 1)

    def _log_queue_put(level=logging.INFO, msg=""):
        try:
            log_queue.put(
                logging.makeLogRecord(
                    {
                        "name": channel_custom_url if channel_custom_url is not None else channel_id,
                        "level": logging._levelToName[level],
                        "levelno": level,
                        "msg": msg,
                    }
                )
            )
        except Exception as e:
            print(f"Error logging: {e}")

    def _log_skip_channel(channel_id, msg):
        _log_queue_put(level=logging.WARNING, msg=msg)
        with open("tmp/skipped_channels.txt", "a") as f:
            f.write(f"{channel_id}|{msg}\n")
    try:
        _log_queue_put(level=logging.INFO, msg=f"Processing channel {channel_id} ({channel_url})")
    except Exception as e:
        print(f"Error logging: {e}")

    try:
        video_ids = yt_get_playlist_ids(channel_url)
        if len(video_ids) < channel_min_videos:
            _log_skip_channel(channel_id, f"Channel {channel_id} has less than {channel_min_videos} videos")
            return all_channel_meta, None

        max_video_idx = min(len(video_ids), n_video_download)
        _skip_premiere_count = 0
        _skip_duration_count = 0
        _continue_for_more_duration_count = 0
        _total_downloaded_duration = 0
        _min_download_duration = max_video_idx * 180 # 3 min * max_video_idx
        # _max_download_duration = max_video_idx * 1800 # 30 min * max_video_idx
        audio_paths = []
        for idx, video_id in enumerate(video_ids):
            try:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_duration = yt_get_video_duration_sec(video_url)
                if video_duration < 180: # 3 min
                    _log_queue_put(level=logging.WARNING, msg=f"Video {video_id} has duration > 3 min, skip")
                    _skip_duration_count += 1
                    continue
                if video_duration > 1800 + 240 + 2: # 30 min + 4 min + 2 sec -> download random 30 min
                    _download_duration = 1800
                    min_ss, max_to = 120, video_duration - 120
                    ss = random.randint(min_ss, max_to - _download_duration)
                    to = ss + _download_duration
                else:
                    # download full - 1 min in the beginning and 1 min in the end
                    ss, to = 60, video_duration - 60
                    _download_duration = to - ss

                audio_path = yt_download_audio(video_url, download_dir, ss=ss, to=to)
                _total_downloaded_duration += _download_duration

                if audio_path is not None:
                    audio_paths.append(audio_path)
            except Exception as e:
                if "PREMIERE_VIDEO" in str(e):
                    max_video_idx += 1
                    _skip_premiere_count += 1
                    continue

            if _skip_premiere_count > 5:
                _log_skip_channel(channel_id, f"Channel {channel_id} has more than 5 premiere videos in the first {max_video_idx} videos")
                # cleanup
                for f in audio_paths:
                    os.remove(f)
                return all_channel_meta, None

            if _skip_duration_count > 5:
                _log_skip_channel(channel_id, f"Channel {channel_id} has more than 5 videos with duration < 3 min in the first {max_video_idx} videos")
                # cleanup
                for f in audio_paths:
                    os.remove(f)
                return all_channel_meta, None

            if idx >= max_video_idx:
                if _continue_for_more_duration_count > 5:
                    break
                if _total_downloaded_duration < _min_download_duration:
                    _continue_for_more_duration_count += 1
                    continue

        segments_path = []
        segments_meta_vad = []
        segments_video_id = []
        for audio_path, video_id in zip(audio_paths, video_ids):
            _segments_path, _segments_meta = vad_split(audio_path, output_dir=segments_dir)
            segments_path.extend(_segments_path)
            segments_meta_vad.extend(_segments_meta)
            segments_video_id.extend([video_id] * len(_segments_path))

            # clean
            os.remove(audio_path)

        segments_snr = [estimate_snr(f) for f in segments_path]

        acss = ac_infer_batch(segments_path)
        torch.cuda.empty_cache()
        speech_probs = ac_get_speech_probs(acss)

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

        # for f in audio_paths:
        #     os.remove(f)

        # save meta
        with open(f"{meta_list_dir}/{channel_id}_all_meta.json", "w", encoding='utf-8') as f:
            f.write(json.dumps(all_channel_meta, indent=4, ensure_ascii=False))
        with open(f"{meta_list_dir}/{channel_id}_selected_meta.json", "w", encoding='utf-8') as f:
            f.write(json.dumps(selected_channel_meta, indent=4, ensure_ascii=False))

        upload_folder_retry(repo_id, "dataset", tmp_dir, path_in_repo=split) # TODO diffrent branch?

        # cleanup
        # remove all tmp_dir
        import shutil
        shutil.rmtree(tmp_dir)

    except Exception as e:
        log_queue.put(logging.makeLogRecord({
            'name': __name__,
            'level': 'ERROR',  # Ensure the level is set correctly
            'levelno': logging.ERROR,
            'msg': f"An error occurred: {e}"
        }))

    return all_channel_meta, selected_channel_meta


def convert_numpy_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)  # Convert numpy integers to native Python int
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)  # Convert numpy floats to native Python float
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert numpy booleans to native Python bool
    elif isinstance(obj, np.void):  # This is a catch-all for other numpy dtypes
        return None
    return obj


def main(df_or_path="tmp/yt_channels.csv", min_snr=20, min_ac_speech_prob=0.9, split="train", repo_id=None, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    all_segments_meta = {}
    selected_channels_meta = {}

    # Create a Manager for shared log queue
    manager = Manager()
    log_queue = manager.Queue()

    # Start log listener in a separate process
    log_listener_process = multiprocessing.Process(target=log_listener, args=(log_queue,))
    log_listener_process.start()

    # Process the data with multiprocessing Pool
    with multiprocessing.get_context('spawn').Pool() as pool:
        results = pool.starmap(process_channel, [(row, min_snr, min_ac_speech_prob, log_queue, repo_id, split) for _, row in df.iterrows()])

    # Stop log listener process
    log_queue.put(None)
    log_listener_process.join()

    # Save the results
    for all_channel_meta, selected_channel_meta in results:
        channel_id = all_channel_meta["id"]
        all_segments_meta[channel_id] = all_channel_meta
        selected_channels_meta[channel_id] = selected_channel_meta

    # Convert all numpy objects within the data to native Python types
    all_segments_meta = json.loads(json.dumps(all_segments_meta, default=convert_numpy_to_native))
    selected_channels_meta = json.loads(json.dumps(selected_channels_meta, default=convert_numpy_to_native))

    # Serialize the results
    with open("tmp/metadata_all.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(all_segments_meta, indent=4, ensure_ascii=False))
    with open("tmp/metadata_selected.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(selected_channels_meta, indent=4, ensure_ascii=False))

    # # Upload to Hugging Face
    # repo_type = "dataset"
    # if not repo_exists(repo_id, repo_type=repo_type):
    #     create_repo(repo_id, repo_type=repo_type, private=True)
    # upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path="tmp", path_in_repo=split)

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
