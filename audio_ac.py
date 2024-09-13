import torch
from transformers import pipeline
from multiprocessing import Lock

cache_dir = "./cache/audio_ac_cache"

pipe = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_dir=cache_dir
)

lock = Lock()


def classify_audio_batch(fpaths):
    with lock:
        predictions = pipe(fpaths, batch_size=8)  # Process in batches of 8
        return predictions
    # results = []
    # for pred in predictions:
    #     scores = {i["label"]: i["score"] for i in pred}
    #     labels = [i["label"] for i in pred]
    #     if "Speech" not in labels:
    #         results.append(False)
    #         continue
    #     if scores["Speech"] < 0.8:
    #         results.append(False)
    #         continue
    #     if list(scores.values())[1] > 0.15:
    #         results.append(False)
    #         continue
    #     results.append(True)
    # return results
