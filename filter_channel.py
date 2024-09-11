from ac import classify_audio_batch
from snr import estimate_snr
from vad import vad_split
from youtube import download_and_cut_n_audio
import librosa  # Ensure this import is here if you're using librosa for loading audio
import os
import torch

def channel_check(url, verbose=False):
    if verbose:
        print(f"start {url}")
    try:
        audio_paths = download_and_cut_n_audio(url, "./step1", max_per_chanel=2)
        if verbose:
            print("N audio paths:", len(audio_paths))
        seg_fpaths_lists = [vad_split(fpath, output_dir="./step2") for fpath in audio_paths]
        if verbose:
            print("N segments:", len(seg_fpaths_lists))
        snrss = [estimate_snr(librosa.load(f)[0]) for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]

        # Classify all segments in batches
        flat_fpaths = [f for seg_fpaths in seg_fpaths_lists for f in seg_fpaths]
        acss = classify_audio_batch(flat_fpaths)
        torch.cuda.empty_cache() # Clear GPU memory
        if verbose:
            print("AC true:", acss.count(True))

        try:
            for f in flat_fpaths:
                os.remove(f)
            for f in audio_paths:
                os.remove(f)
        except Exception as e:
            print(e)
        
        print(f"Final result: {url}")
        with open("out.txt", "a") as f:
            f.write(str({
                "url": url,
                "snrss": snrss,
                "acss": acss,
            }) + "\n")
    except Exception as e:
        print(f"Error: {e}")

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
    import sys
    import multiprocessing

    with open(sys.argv[1], "r") as f:
        channel_urls = [l.strip() for l in f.readlines()]

    with multiprocessing.get_context('spawn').Pool(2) as pool:
        pool.map(channel_check, channel_urls)
