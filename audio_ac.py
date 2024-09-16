from multiprocessing import Lock
from typing import List

import librosa
import torch
from datasets import Dataset
from transformers import pipeline, ASTFeatureExtractor, AutoModelForAudioClassification

cache_dir = "./cache/audio_ac_cache"

pipe = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_dir=cache_dir
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feature_extractor = ASTFeatureExtractor()
# model = AutoModelForAudioClassification.from_pretrained(
#     "MIT/ast-finetuned-audioset-10-10-0.4593",
#     cache_dir=cache_dir,
# )
# model = model.to(device)


lock = Lock()


def classify_audio_batch(fpaths: List[str]):
    with lock:
        predictions = pipe(fpaths)
        # audio_data = [librosa.load(fpath, sr=16000)[0] for fpath in fpaths]
        # inputs = feature_extractor(audio_data, sampling_rate=16000, padding="max_length", return_tensors="pt")
        # input_values = inputs.input_values.to(device)
        # with torch.no_grad():
        #     outputs = model(input_values)

        # # post-process the output like pipeline
        # logits = outputs.logits
        # predictions = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().tolist()

        return predictions


if __name__ == "__main__":
    results = classify_audio_batch([
        'data/audio_sample/clean/HOHDaZ3BLxY_00000074.wav',
        'data/audio_sample/clean/p232_258.wav',
    ])
    print(results)
