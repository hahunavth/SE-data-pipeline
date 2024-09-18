import json
import logging
import math
import multiprocessing
import os
from multiprocessing import Manager

import librosa
import numpy as np
import pandas as pd
import torch
from huggingface_hub import create_repo, repo_exists, upload_folder
from tqdm import tqdm

from audio_ac import ac_infer_batch, ac_get_speech_probs
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import yt_download_audio, yt_get_video_duration_sec, yt_get_playlist_ids


# Global variables for directories
download_dir = "tmp/downloaded"
segments_dir = "tmp/segments"

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers for logging
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("tmp/pipeline.log")

# Set log levels
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.INFO)

# Create formatters and add to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


def channel_check(args):
    i, url, verbose, clean_step_1, clean_step_2 = args
    # if verbose:
    logger.debug(f"start[{i}]: {url}")

    try:
        pass


    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        with open("channel_check_err.jsonl", "a") as f:
            f.write(json.dumps({
                "url": url,
                "error": str(e).replace("\n", "\t"),
            }) + "\n")

def process_channel(row, min_snr, min_ac_speech_prob, log_queue, channel_min_videos=5):
    all_channel_meta = {**row}
    all_channel_meta["videos"] = {}
    all_channel_meta["selected"] = True
    selected_channel_meta = {**row}
    selected_channel_meta["videos"] = {}

    channel_id = row["id"]
    channel_url = row["url"]
    channel_n_subs = row["n_subs"]
    channel_n_videos = row["n_videos"]

    def _log_queue_put(level=logging.INFO, msg=""):
        try:
            log_queue.put(
                logging.makeLogRecord(
                    {
                        "name": __name__,
                        "level": logging._levelToName[level],
                        "levelno": level,
                        "msg": msg,
                    }
                )
            )
        except Exception as e:
            print(f"Error logging: {e}")

    _log_queue_put(
        logging.INFO,
        f"Processing channel {channel_id} ({channel_url}) with {channel_n_subs} subscribers",
    )

    try:
        # Skip if channel has less than `channel_min_videos` videos
        if channel_n_videos < channel_min_videos:
            _log_queue_put(
                logging.INFO,
                f"Channel {channel_id} has less than {channel_min_videos} videos, skipping",
            )
            msg = f"n_videos={channel_n_videos} is less than {channel_min_videos}"
            with open("tmp/skipped_channel.txt", "a") as f:
                f.write(f"{channel_id}|{msg}\n")
            all_channel_meta["selected"] = False
            all_channel_meta["skip_reason"] = msg
            return all_channel_meta, None

        video_ids = yt_get_playlist_ids(channel_url)

        # filter channel



        # Skip if n_video is less than 10
    except Exception as e:
        pass
