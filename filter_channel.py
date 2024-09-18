import json
import multiprocessing
import os
import sys

import librosa
import torch

from audio_ac import ac_infer_batch
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import download_and_cut_n_audio


def channel_check(args):
    # url, verbose=False, clean_step_1=True, clean_step_2=True
    i, url, verbose, max_per_chanel, clean_step_1, clean_step_2 = args
    if verbose:
        print(f"start[{i}]: {url}")
    try:
        audio_paths = download_and_cut_n_audio(url, "./step1", max_per_chanel=max_per_chanel)
        if verbose:
            print("N audio paths:", len(audio_paths))
        seg_fpaths_lists = [vad_split(fpath, output_dir="./step2")[0] for fpath in audio_paths]
        if verbose:
            print("N segments:", len(seg_fpaths_lists))
        snrss = [estimate_snr(f) for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]

        # Classify all segments in batches
        flat_fpaths = [f for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]
        acss = ac_infer_batch(flat_fpaths)
        torch.cuda.empty_cache() # Clear GPU memory
        if verbose:
            print("AC true:", acss.count(True))

        try:
            if clean_step_1:
                for f in audio_paths:
                    os.remove(f)
                
            if clean_step_2:
                for f in flat_fpaths:
                    os.remove(f)
        except Exception as e:
            print(e)
        
        print(f"Final result: {url}")
        with open("out.jsonl", "a") as f:
            f.write(json.dumps({
                "url": url,
                "snrss": snrss,
                "acss": acss,
            }) + "\n")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        with open("err.jsonl", "a") as f:
            f.write(json.dumps({
                "url": url,
                "error": str(e).replace("\n", " "),
            }) + "\n")


def test_check_audio(video_url="https://www.youtube.com/watch?v=CgPHMBWzgiY"):
    from yt_download import yt_download_audio
    output_dir = "./tmp/test_check_audio"

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading audio")
    audio_path = yt_download_audio(video_url, os.path.join(output_dir, "step1.1"), print_err=True)
    audio_paths = [audio_path]

    print("VAD splitting")
    seg_fpaths_lists = [vad_split(
        fpath, 
        output_dir=os.path.join(output_dir, "step2"), 
        min_silence_duration_ms=500,
        speech_pad_ms=30,
    )[0] for fpath in audio_paths]

    print("Estimating SNR")
    snrss = [estimate_snr(f) for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]

    print("Classifying audio")
    flat_fpaths = [f for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]
    acss = ac_infer_batch(flat_fpaths)

    with open(os.path.join(output_dir, "out.json"), "a") as f:
        f.write(json.dumps({
            "url": video_url,
            "snrss": snrss,
            "acss": acss,
        }) + "\n")


def main(fpath, n_process=2, verbose=False, max_per_chanel=2, no_clean_step_1=False, no_clean_step_2=False):
    with open(fpath, "r") as f:
        channel_urls = [l.strip() for l in f.readlines()]

    with multiprocessing.get_context('spawn').Pool(n_process) as pool:
        pool.map(channel_check, [(i, url, verbose, max_per_chanel, not no_clean_step_1, not no_clean_step_2) for i, url in enumerate(channel_urls)])



if __name__ == '__main__':
    # channel_urls = [
    #     'https://www.youtube.com/@MixiGaming3con', 
    #     "https://www.youtube.com/@Optimus96", 
    #     "https://www.youtube.com/@duyluandethuong",
    #     "https://www.youtube.com/c/MuseVi%E1%BB%87tNam",
    #     "https://www.youtube.com/@VFacts",
    #     "https://www.youtube.com/@bonao",
    #     "https://www.youtube.com/@MacOnevn",
    # ]
    from fire import Fire
    Fire(main)