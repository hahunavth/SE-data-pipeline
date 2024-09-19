import json
import logging
import math
import multiprocessing
import os
import random
import time
from multiprocessing import Manager
import shutil

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from audio_ac import ac_get_speech_probs, ac_infer_batch
from audio_snr import estimate_snr
from audio_vad import vad_split
from hf import upload_folder_retry
from yt_download import (yt_download_audio, yt_get_playlist_ids,
                         yt_get_video_duration_sec)

from pydub import AudioSegment


added_video_ids_set = set([
    "MVp0-wpyd8o",
    "gH66YiIUSJg",
    "Bylj5qWbC-g",
    "ouubBo_PD_4",
    "9PPUXwrr3Vk",
    "PpcSSyYHhfQ",
    "Vm62VF-tbXo",
    "fuhTUoXNrcs",
    "mNB6V1BqIKk",
    "in-jwdVAzAQ",
    "Ssjw31blRms",
    "bLocj277ccg",
    "I3TWGzoSwVc",
    "ku5u1frJ5ig",
    "lnV3ql9vl6M",
    "YBmd-uj8itY",
    "af9IbQZgY-s",
    "0FrMteTWKfY",
    "SF-nWbwKFCM",
    "jDNukRpCSe4",
    "2iEySRrFhek",
    "dV5bno0Spqo",
    "SY_QHJMloN4",
    "CCVvuf8GOsU",
    "4cOtDHI1EKA",
    "yixlauWu-X4",
    "g0uUCJDAnjM",
    "4e28CZbB3Yw",
    "0CQ1Y-19Ol8",
    "26jmaOS-hs8",
    "Km3ZbHFZZMw",
    "uv2_LA8LbCQ",
    "6oMfKjt-DL0",
    "PTX9u-bRyH4",
    "sSqhAPw4N2Y",
    "wkkU1CM4oLA",
    "mf7fAYYDP3A",
    "FXeP4HgkG0I",
    "azwJg2CzTVQ",
    "ql1BL3n7gJU",
    "MvREbKZNdFQ",
    "_8a7QhtzCVA",
    "j6Zrem7wut0",
    "7zPPzd_GvT8",
    "ku_DROqhGI0",
    "rbyRJSfOH30",
    "kFYs2iCK_HA",
    "oO9zmWUlDEg",
    "k8bhs3zzABs",
    "5y9AOdMC_kI",
    "aZX-Hk3IqFU",
    "fXee9OusLJc",
    "N-tM2TGWRB4",
    "c0r9hrhYr7g",
    "z7u0RhXwXSw",
    "cD6gLZNEKWM",
    "bxd0b-lc8Qw",
    "9gAILwQ-LvY",
    "pzuUywCyUyg",
    "IWeflXrQIj0",
    "0_flsldlBPc",
    "eb1i3dgDDMg",
    "3acrNLvMCT8",
    "AjZRIdCmQXA",
    "anFyPyn-ygw",
    "BNfZoky_8Nc",
    "cDrgRH0aACs",
    "z42zl9RiGSc",
    "oFax1_QCXJE",
    "zvakFEwxJRE",
    "JWism0YMJpA",
    "HlUp3XT-0RQ",
    "8npaugFg-Bk",
    "hcJxk__aGsc",
    "YbHyplBi7cA",
    "n1UNKkfT-x4",
    "7yrd2ROhtbs",
    "jXUy87qsIZo",
    "DULrc0hl604",
    "IKOkZSUB71I",
    "fuNIDdCbVmw",
    "RJN8K5EGlX0",
    "rsEVv7PIc3Q",
    "1GRmo62PNOY",
    "ry15JbtQp2Q",
    "hg5JMPvvdz4",
    "kEoKxuAPeNk",
    "5y42sycWYmI",
    "p221-6BWdaU",
    "xW8qJXtWzpk",
    "540xjBxJsrE",
    "IJwCJZNVZWI",
    "266tgHfjopM",
    "gtyU4sT82hM",
    "FZXTDWwWQVQ",
    "pEhXPreMJB0",
    "qU0TP8nwXGs",
    "0O1UeyzbNlk",
    "8qxcF3Zwywc",
    "EyaKgEYCOzo",
    "lrxAf7FsYh0",
    "yRvFFSSBksg",
    "_EfZbWcgUHM",
    "0pcs1iiRbwM",
    "Yx5CzpLb0Xo",
    "B-9xoZ43RRg",
    "y7jsgwpgVGI",
    "H4UPQhx7ANU",
    "G9NSl60f9mE",
    "jImLM5XoqYw",
    "OsaDBqHskKI",
    "qAN6r21EIF4",
    "1_jH06pNsR8",
    "2p85pekNWxo",
    "DVkibuHM7Gc",
    "h0ffQUWZfoA",
    "IQ3l6DJ_Xgw",
    "zhuqbxOO0Bo",
    "96RiVBzMZDw",
    "GdRX-ebJ7pU",
    "NUGb8aqFD4A",
    "fddmAap63Pk",
    "NyP7Kgl9R2w",
    "L_Wy3Q8ZU_8",
    "HiyQxKdI5So",
    "RnWKj55mq4M",
    "qgCxKXw-igk",
    "1ggYDp4XSHU",
    "gdNwOznxaTU",
    "d-COkwzcknQ",
    "pf27Wv5WGrg",
    "r_4T6fZH_t0",
    "ECn3Stp4eSc",
    "gRXJnKin6gM",
    "7c0EvfTPikM",
    "3eqcnF5Vdnk",
    "giDW2tPB84U",
    "HURm2aqLrSc",
    "j8MatWSQNPU",
    "ULAM_PzSnWg",
    "NdLRfH-y_Wc",
    "MXKd9JV-ELM",
    "yo53QonPPXA",
    "JjnPLbKXgQE",
    "JOZ97Hb-htU",
    "MkiJKp3B0cc",
    "-mA8GXioqsA",
    "GDtorGf09mc",
    "uSUU8QbCUoc",
    "j9RKAiebLA8",
    "GhMyqXpSzAs",
    "NHjlqkHJZ1Y",
    "Gcd-2JA2TLg",
    "REIgAuianYk",
    "x6j8j16BHKM",
    "OvQhjDQ9qak",
    "MiChZPaGfJM",
    "9yphKX3Vl3k",
    "0Q_PPgwGS2M",
    "_RDRa-BP95Y",
    "fBlBMu9fZQI",
    "fWKacFMthSQ",
    "da1mAMF4_IE",
    "crBxZhS-nBw",
    "bKNLo0fJN78",
    "D4DxhWwzfKk",
    "Hcw_wWAZvuo",
    "gaUZGMwYDwo",
    "NLqu6vqZzgg",
    "5jp25WXMh18",
    "WfKNeewwBAM",
    "eaySEGBVfQw",
    "rwQ_H3m8OR0",
    "P1BPQCYqtjI",
    "XQLNiDSdeso",
    "ndFvsjP6Hrw",
    "Wggnd6GwAmw",
    "vRicUXwdW10",
    "vwYBMbnzFpY",
    "8rTkhSTqT5Y",
    "x3DvWsTF50o",
    "OQhZ-d_I9pk",
    "09d-BXwQ-f8",
    "OKxlrI-bFHs",
    "oHvxdK9Eym8",
    "2Ukd7x44CjM",
    "nBdET-Sd7pI",
    "l3I4X6yzCVg",
    "uHOfNlhnDwg",
    "DvVffrx3nDQ",
    "IfcqvRuVLZc",
    "s_FrwcfXxYA",
    "JcKDU9AG5Io",
    "0dOjjIqsB0A",
    "s8VLLkqTe7M",
    "tMxh5OZuhzQ",
    "fXX6gFwq3tk",
    "ap9qy1p2clQ",
    "0_gai2buQeA",
    "t2U_27pEzRI",
    "3AxD76TBZQY",
    "Z0VsO3FRTNQ",
    "IS3QLF79Ek4",
    "CkQbt_MPipA",
    "WljSbNHDgNw",
    "i3-yVolKFio",
    "4IpVZI-sMoc",
    "mCTj0Gz4Hc8",
    "-bZN3o6sHwY",
    "pnncnSqDCPo",
    "zec9E-wJvhk",
    "SI-pAof_6vg",
    "0IO6bY2woMw",
    "rf68Q5VP26I",
    "ag30J8yq0iE",
    "HN8zBrpV1Pc",
    "u09BoXfa0Wk",
    "iQsj6YfZZk4",
    "uhDLzb7rL1k",
    "4Iitbt-6M68",
    "kxm_OQFmsFc",
    "fbIlIwyedjk",
    "0sYeaYFikps",
    "Sr9Nheg4CGw",
    "vaHtJNue5PI",
    "v7KcBUdcayM",
    "M6yWmTPyk9I",
    "J9i6mZHLx2I",
    "PHyT7mcPBfk",
    "SsgDFfzyOW0",
    "gvzmBZJkiHE",
    "76EnmaHRkUo",
    "1g1SNchgfFs",
    "l-PqqHVc9H8",
    "E3IiWG8NFpQ",
    "zXUwa5CARnI",
    "3zD4pWvMke4",
    "GSO9-RSUw1g",
    "NVM6tV3RbFE",
    "F-0GQlUSfmA",
    "3sEP_UezPUo",
    "zyCg-zzY3lQ",
    "pfX-DbRMqRA",
    "NF7pTIHof5k",
    "eGFg0AqvegI",
    "JbtzXh8U9EA",
    "6lOLrU6oJyc",
    "pwrkodQZUSM",
    "xsnrd4U5Ryk",
    "NgqbkTrkeBI",
    "G5yTJA8djDY",
    "1y4q8oVvmPM",
    "KM7r_h0gNeI",
    "FqzIQhOZ_N4",
    "SSeNDpRxj9M",
    "UN60UNK2iiQ",
    "VLZoxZHPmKA",
    "hG7A_yLb7no",
    "g0QHLerAxg0",
    "Ab8x2w9LXKM",
    "2L0EniSXNiE",
    "u6dUNEZ4xK8",
    "PrYR94c7IXc",
    "Y3AaMoNie2k",
    "TVxN2Gb6lEc",
    "rsmwLE-QvUY",
    "ZPPl8LZdmI4",
    "ml8ZLniGX_o",
    "qtm0Hm8jttQ",
    "lFdn6vzUuS0",
    "qH1Ck7jzDsA",
    "rIzjczL__jQ",
    "XTuNEa68LXQ",
    "_7gGoCpBRUg",
    "aREKS4UZyGM",
    "QFaNk-Uz6rc",
    "8n0bJF2Zxrc",
    "d6cbEVpoZyg",
    "-X6wkgLYt30",
    "I0GHxT-XGEQ",
    "BRfptCRHWXc",
    "umPmyMJikdw",
    "HaVYwD02aU8",
    "PFkVfB5t4Lk",
    "rfjFXhAJlTg",
    "BfSivwI9rkE",
    "p2TgxdBQpi4",
    "w10zGmXvxOc",
    "D0CAy4rxVMM",
    "dlwg3WjQko4",
    "dV71HKIC4Dg",
    "6JWE9Ncmcwg",
    "TXldLPEOhCk",
    "WjzSTv_-Z4I",
    "AhJkaE63cU8",
    "CgDpNyxHtfY",
    "acojgwTHL70",
    "-O2xwYwYfR8",
    "Afrm6DIcTMU",
    "i6ZUSLovgp4",
    "GW6EbHdo-uY",
    "V1Xl0rrA52U",
    "nhWlc4U7vK0",
    "BlMRFfedWz4",
    "vQVFCEZ7iBg",
    "hvrWPTaHXv8",
    "A1EqZLBnYCY",
    "NRTJun00AiY",
    "jqyCH5bV74Y",
    "f-HsFW86SWk",
    "pm1rkfRovRo",
    "OtYCQ1Z214c",
    "Tv8XkaFIl_8",
    "5epmRpg7AUY",
    "Jy-Y5n8k-g8",
    "tEeEWCOMo1E",
    "lZRgtppUr50",
    "qH-9XIJBpaY",
    "fgljFZZ2xes",
    "nMy17eB2Zfc",
    "CDDpkD9M7-s",
    "DzKn1vEBq_s",
    "zG1-GGD6eDE",
    "r5LSV6NRQv0",
    "DSZKM1_yTgg",
    "QVSqDa4hqbc",
    "UAO4oJvb5JE",
    "1cm-eudw0eA",
    "xjU0FWdAhcE",
    "Tk7EuxnmSpk",
    "FxwawNrcwr4",
    "glj7JXUyLZU",
    "XnQga7gr5yQ",
    "GGcwCrBWA7M",
    "LSNYaaVsO0o",
    "zLTEGuorXTE",
    "gKfpjtndqZc",
    "9PPUXwrr3Vk",
    "PpcSSyYHhfQ",
    "Vm62VF-tbXo",
    "fuhTUoXNrcs",
    "mNB6V1BqIKk",
    "in-jwdVAzAQ",
    "Z1NZwePLz9s",
    "w0z09gSpaLY",
    "zF0uq-ggdSw",
    "1xNOFJVzP80",
    "_fwqmu9FKnU",
    "_CU2-YVufr4",
    "6DRdwO2fgwM",
    "XRMz1T9GSjs",
    "yn3K1OuOwCk",
    "IPOz78HO8no",
    "vMFi9i9H_Q0",
    "jD6nGv9QV9I",
    "8PqNRUl6uSs",
    "twRwVcWsdAg",
    "BGlE5byLxPA",
    "IMHVOEg7BB8",
    "RZL_Ch6GUM8",
    "DfvS2PIdGkQ",
    "EuKTz-mRzZo",
    "hFu2qlxc-CA",
    "oEsvrrcWbnA",
    "omTihLzD91k",
    "Mv4u7MYqJlA",
    "k3Gt6Icve8I",
    "aK_bxOQh4aw",
    "jJCk2rcz6AU",
    "E61Z8K-uKzM",
    "4SNObGN8--8",
    "GuM_jlSaBYM",
    "tvezpkpy1bM",
    "7Mk_yTfq9Zc",
    "mub_XmWwAvk",
    "XtBTC5mVoIU",
    "Q9RIx1pix3M",
    "kARJECeILgU",
    "oomoWM-awyM",
    "tGMv26OgxLM",
    "jXUfdgGskMo",
    "Ngp4IdO2jm4",
    "Br4roY5EC6U",
    "FiegNlMpJAo",
    "Jh3c9O3N9gg",
    "b4oVrZVk_TE",
    "g1nA9-Pawn0",
    "gn43J-KEhSU",
    "RSAbUNd545c",
    "0vosxhxS5-Y",
    "iq21qB3Rcp8",
    "y-1oaUvoQOU",
    "mD8lhWkJh3w",
    "iHpTf-ni_lc",
    "w24FvAZWEjU",
    "skUqkxVH6hM",
    "VPNW9MwtC9w",
    "yhjlmAMBo6Q",
    "HiWnbFZqHVY",
    "ikzc0VjSLoc",
    "nR1-bWDWGxA",
    "fri26Bu1nS4",
    "nJkOaUut0ZM",
    "dylqVVJtNbI",
    "dDHTjr6oOJU",
    "GCh94akcjjA",
    "4w6h-SRgs1g",
    "LSkYppcH1Yg",
    "m3NYSNsC48Q",
    "oQyWdsqTCdc",
    "wsm6iCHgM4c",
])


def replace_with_cutted_audio(audio_path, ss, to):
    if ss is None and to is None:
        return audio_path
    audio = AudioSegment.from_wav(audio_path)
    cutted_audio = audio[ss * 1000:to * 1000]
    os.remove(audio_path)
    cutted_audio.export(audio_path, format="wav")
    return audio_path


def get_process_dirs(idx):
    tmp_dir = f"tmp/{idx}"
    download_dir = f"tmp/downloaded"
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


def process_channel(idx, row, min_snr, min_ac_speech_prob, log_queue, repo_id=None, split="test", branch="main", channel_min_videos=5):

    time.sleep(random.randint(idx, idx * 3))

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

    if channel_n_sub < 10000:
        n_video_download = 5
    elif channel_n_sub < 30000:
        n_video_download = 10
    elif channel_n_sub < 50000:
        n_video_download = 25
    elif channel_n_sub < 100000:
        n_video_download = 40
    elif channel_n_sub < 200000:
        n_video_download = 50
    else:
        n_video_download = 60

    def _log_queue_put(level=logging.INFO, msg=""):
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

    def _log_skip_channel(channel_id, msg):
        _log_queue_put(level=logging.INFO, msg=f"SKIP: {msg}")
        with open("tmp/skipped_channels.txt", "a") as f:
            f.write(f"{channel_id}|{msg}\n")

    _log_queue_put(level=logging.INFO, msg=f"Processing channel {channel_id} ({channel_url})")

    try:
        if channel_n_videos < channel_min_videos:
            _log_skip_channel(channel_id, f"Channel {channel_id} has less than {channel_min_videos} videos")
            return all_channel_meta, None

        video_ids = yt_get_playlist_ids(channel_url)

        n_ignore = len(added_video_ids_set.intersection(video_ids))

        max_video_idx = min(len(video_ids), n_video_download) - n_ignore
        _skip_premiere_count = 0
        _skip_duration_count = 0
        _continue_for_more_duration_count = 0
        _total_downloaded_duration = 0
        channel_total_duration_h = 0
        to_upload_duration_h = 0
        n_video_downloaded = 0

        _log_queue_put(msg=f"Max download {max_video_idx} videos")

        _min_download_duration = (max_video_idx - n_ignore) * 180 # 3 min * max_video_idx
        # _max_download_duration = max_video_idx * 1800 # 30 min * max_video_idx
        _uploaded = False
        for v_idx, video_id in enumerate(video_ids):
            if video_id in added_video_ids_set:
                continue

            if _skip_premiere_count > 3:
                _log_skip_channel(channel_id, f"Channel {channel_id} has more than 3 premiere videos in the first {max_video_idx} videos")
                # return all_channel_meta, None
                break

            if _skip_duration_count > 3:
                _log_skip_channel(channel_id, f"Channel {channel_id} has more than 3 videos with duration < 3 min in the first {max_video_idx} videos")
                # return all_channel_meta, None
                break

            if _continue_for_more_duration_count > 3:
                _log_skip_channel(channel_id, f"Channel {channel_id} has _continue_for_more_duration_count > 3")
                break
            _log_queue_put(msg=f"_continue_for_more_duration_count {_continue_for_more_duration_count}")
            if v_idx >= max_video_idx:
                if _total_downloaded_duration < _min_download_duration:
                    _continue_for_more_duration_count += 1
                    continue
                else:
                    break

            _uploaded = False
            try:
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                _log_queue_put(level=logging.INFO, msg=f"Downloading video {v_idx} {video_url}")
                ss, to = None, None
                audio_path = yt_download_audio(video_url, download_dir)
                _download_duration = math.floor(librosa.get_duration(path=audio_path))

                if _download_duration < 180: # 3 min
                    ss, to = None, None
                    pass
                elif _download_duration > 1800 + 240 + 2: # 30 min + 4 min + 2 sec -> download random 30 min
                    # cut this video to 
                    ss, to = 120, _download_duration - 120
                    _download_duration = to - ss
                else:
                    # download full - 1 min in the beginning and 1 min in the end
                    ss, to = 60, _download_duration - 60
                    _download_duration = to - ss
                replace_with_cutted_audio(audio_path, ss, to)

                _total_downloaded_duration += _download_duration

                if audio_path is None:
                    # channel_audio_paths.append(audio_path)
                    _log_queue_put(msg="SKIP_AUDIO_PATH_NONE")
                    continue
            except Exception as e:
                if "PREMIERE_VIDEO" in str(e) or "OFFLINE_VIDEO" in str(e):
                    max_video_idx += 1
                    _skip_premiere_count += 1
                    continue
                import traceback
                traceback.print_exc()
                raise e

            _log_queue_put(msg="VAD")
            os.makedirs(segments_dir, exist_ok=True)
            segments_path, segments_meta = vad_split(audio_path, output_dir=segments_dir)

            # clean audio_path
            os.remove(audio_path)

            n_video_downloaded += 1

            # SNR
            _log_queue_put(msg="SNR")
            segments_snr = [estimate_snr(f) for f in segments_path]

            # AC
            _log_queue_put(msg="AC")
            acss = ac_infer_batch(segments_path)
            torch.cuda.empty_cache()
            speech_probs = ac_get_speech_probs(acss)

            _log_queue_put(msg="META")
            # Add to all_channel_meta and selected_channel_meta
            for seg_idx, (f, vad_meta, snr, speech_prob, acs) in enumerate(zip(segments_path, segments_meta, segments_snr, speech_probs, acss)):
                is_selected = snr >= min_snr and speech_prob >= min_ac_speech_prob
                embed_url = f"https://www.youtube.com/embed/{video_id}?start={math.floor(vad_meta['start'] / 16000)}&end={math.ceil(vad_meta['end'] / 16000)}"
                all_channel_meta["videos"].setdefault(video_id, []).append({
                    "idx": os.path.basename(f).replace(".wav", ""),
                    "url": embed_url,
                    "selected": is_selected,
                    "vad": vad_meta,
                    "snr": snr,
                    "ac": acs,
                })

                if is_selected:
                    selected_channel_meta["videos"].setdefault(video_id, []).append({
                        "idx": os.path.basename(f).replace(".wav", ""),
                        "url": embed_url,
                        "start": vad_meta["start"],
                        "end": vad_meta["end"],
                    })
                    channel_total_duration_h += (vad_meta["start"] - vad_meta["end"]) / 16000 / 3600
                    to_upload_duration_h += (vad_meta["start"] - vad_meta["end"]) / 16000 / 3600
                else:
                    os.remove(f)

            if (v_idx != 0 and v_idx % 30 == 0):
                _log_queue_put(msg="PUSH_TO_HUB")
                _uploaded = True
                # save meta
                _all_channel_meta = json.loads(json.dumps(all_channel_meta, default=convert_numpy_to_native))
                _selected_channel_meta = json.loads(json.dumps(selected_channel_meta, default=convert_numpy_to_native))
                with open(f"{meta_list_dir}/{channel_id}_all_meta.json", "w", encoding='utf-8') as f:
                    f.write(json.dumps(_all_channel_meta, indent=4, ensure_ascii=False))
                with open(f"{meta_list_dir}/{channel_id}_selected_meta.json", "w", encoding='utf-8') as f:
                    f.write(json.dumps(_selected_channel_meta, indent=4, ensure_ascii=False))
                # upload hf
                upload_folder_retry(repo_id, "dataset", tmp_dir, path_in_repo=split, revision=branch, commit_message=f"[{channel_custom_url if channel_custom_url is not None else channel_id}]({v_idx}) upload {to_upload_duration_h}h")
                # remove all segments_path
                shutil.rmtree(segments_dir)
                to_upload_duration_h = 0

            # END VIDEO LOOP

        # END LOOP

        if not _uploaded:
            # save meta
            _all_channel_meta = json.loads(json.dumps(all_channel_meta, default=convert_numpy_to_native))
            _selected_channel_meta = json.loads(json.dumps(selected_channel_meta, default=convert_numpy_to_native))
            with open(f"{meta_list_dir}/{channel_id}_all_meta.json", "w", encoding='utf-8') as f:
                f.write(json.dumps(_all_channel_meta, indent=4, ensure_ascii=False))
            with open(f"{meta_list_dir}/{channel_id}_selected_meta.json", "w", encoding='utf-8') as f:
                f.write(json.dumps(_selected_channel_meta, indent=4, ensure_ascii=False))
            # upload hf
            upload_folder_retry(repo_id, "dataset", tmp_dir, path_in_repo=split, revision=branch, commit_message=f"[{channel_custom_url if channel_custom_url is not None else channel_id}](END) upload {to_upload_duration_h}h")

        _log_queue_put(msg=F"DONE: {channel_id}, total_duration={channel_total_duration_h}h, n_video={n_video_downloaded}")

    except Exception as e:
        log_queue.put(logging.makeLogRecord({
            'name': __name__,
            'level': 'ERROR',  # Ensure the level is set correctly
            'levelno': logging.INFO,
            'msg': f"An error occurred: {e}"
        }))
    finally:
        # cleanup
        # remove all tmp_dir
        shutil.rmtree(tmp_dir)

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


def main(df_or_path="tmp/yt_channels.csv", min_snr=20, min_ac_speech_prob=0.9, split="train", branch=None, repo_id=None, verbose=False, num_workers=4):

    if branch is None:
        branch = "main"
    elif branch != "main":
        from huggingface_hub import create_branch
        create_branch(repo_id, branch=branch, revision="224336fadb8572137ad77541e7f6550d1f11f3db", repo_type="dataset", exist_ok=True)

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
    with multiprocessing.get_context('spawn').Pool(num_workers) as pool:
        results = pool.starmap(process_channel, [(idx, row, min_snr, min_ac_speech_prob, log_queue, repo_id, split, branch) for idx, row in df.iterrows()])

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
