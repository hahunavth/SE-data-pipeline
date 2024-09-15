import json
import multiprocessing
import os
import sys

import librosa
import torch

from audio_ac import classify_audio_batch
from audio_snr import estimate_snr
from audio_vad import vad_split
from yt_download import download_and_cut_n_audio


def channel_check(args):
    # url, verbose=False, clean_step_1=True, clean_step_2=True
    i, url, verbose, clean_step_1, clean_step_2 = args
    if verbose:
        print(f"start[{i}]: {url}")
    try:
        audio_paths = download_and_cut_n_audio(url, "./step1", max_per_chanel=2)
        if verbose:
            print("N audio paths:", len(audio_paths))
        seg_fpaths_lists = [vad_split(fpath, output_dir="./step2")[0] for fpath in audio_paths]
        if verbose:
            print("N segments:", len(seg_fpaths_lists))
        snrss = [estimate_snr(f) for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]

        # Classify all segments in batches
        flat_fpaths = [f for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]
        acss = classify_audio_batch(flat_fpaths)
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


def main(fpath, n_process=2, verbose=False, no_clean_step_1=False, no_clean_step_2=False):
    with open(fpath, "r") as f:
        channel_urls = [l.strip() for l in f.readlines()]

    with multiprocessing.get_context('spawn').Pool(n_process) as pool:
        pool.map(channel_check, [(i, url, verbose, not no_clean_step_1, not no_clean_step_2) for i, url in enumerate(channel_urls)])



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