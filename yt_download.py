import os
import re
import subprocess
import time

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.io.wavfile import write
from tqdm import tqdm


def get_youtube_playlist_ids(channel_url, print_err=True):
    try:
        command = ["yt-dlp", "--flat-playlist", "--print", "id", channel_url]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            if print_err:
                print(f"Error: {result.stderr}")
            return None
        ids = result.stdout.strip().split("\n")
        return [ids for ids in ids if ids]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def check_audio_quality_48k(url, retry_if_error=True):
    try:
        result = subprocess.run(["yt-dlp", "-F", url], capture_output=True, text=True)

        # if error fetching formats
        if result.returncode != 0:
            # print(f"Error fetching formats: {result.stderr}")
            return False

        output = result.stdout
        is_req_valid = False
        for line in output.splitlines():
            if "audio only" in line:
                is_req_valid = True
            if "audio only" in line and ("48000Hz" in line or "48k" in line):
                # print(f"Found 48kHz audio: {line}")
                return True
        if not is_req_valid:
            time.sleep(0.5)
            if retry_if_error:
                return check_audio_quality_48k(url, retry_if_error=False)
            else:
                return False

        # If no 48kHz audio was found
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def download_audio(video_url, output_dir="./", print_err=True):
    #     try:
    os.makedirs(output_dir, exist_ok=True)
    video_id = video_url.split("v=")[-1]
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    command = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        "-ar 48000",
        "-o",
        output_template,
        video_url,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        if print_err:
            print(f"Error: {result.stderr}")
        return None

    output_file = os.path.join(output_dir, f"{video_id}.wav")
    if not os.path.exists(output_file):
        return None

    return output_file


def cut_audio_to_10_minutes(
    audio_path, output_dir="./", prefix="", remove_input_file=True
):
    #     try:
    audio = AudioSegment.from_wav(audio_path)

    start = len(audio) // 2 - 300000
    if start > 0:
        ten_min_audio = audio[start : start + 600000]
    else:
        ten_min_audio = audio[:600000]

    filename = os.path.basename(audio_path)
    output_path = os.path.join(output_dir, f"{prefix}{filename}")

    ten_min_audio.export(output_path, format="wav")

    if remove_input_file:
        os.remove(audio_path)

    return output_path


#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


def download_and_cut_n_audio(channel_url, output_dir="./", max_per_chanel=2):
    os.makedirs(os.path.join(output_dir, "step1.1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "step1.2"), exist_ok=True)
    try:
        video_ids = get_youtube_playlist_ids(channel_url)[:max_per_chanel]
        if not video_ids:
            return []

        audio_paths = []

        for video_id in video_ids:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            if not check_audio_quality_48k(video_url):
                raise Exception(f"Audio quality is not 48kHz: {video_url}")
            audio_path = download_audio(video_url, os.path.join(output_dir, "step1.1"))
            if audio_path:
                audio_path = cut_audio_to_10_minutes(
                    audio_path, os.path.join(output_dir, "step1.2")
                )
                audio_paths.append(audio_path)

        return audio_paths
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise e # test
