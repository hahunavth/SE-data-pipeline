from typing import List

import librosa
import torch
from datasets import Dataset
from transformers import pipeline, ASTFeatureExtractor, AutoModelForAudioClassification

from mp import gpu_lock, device


AC_CACHE_DIR = "./cache/audio_ac"


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feature_extractor = ASTFeatureExtractor()
# model = AutoModelForAudioClassification.from_pretrained(
#     "MIT/ast-finetuned-audioset-10-10-0.4593",
#     cache_dir=cache_dir,
# )
# model = model.to(device)

# def ac_infer_batch(fpaths: List[str]):
#     with gpu_lock:
#         audio_data = [librosa.load(fpath, sr=16000)[0] for fpath in fpaths]
#         inputs = feature_extractor(audio_data, sampling_rate=16000, padding="max_length", return_tensors="pt")
#         input_values = inputs.input_values.to(device)
#         with torch.no_grad():
#             outputs = model(input_values)

#         # post-process the output like pipeline
#         logits = outputs.logits
#         predictions = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().tolist()


pipe = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device=device,
    cache_dir=AC_CACHE_DIR,  # FIXME: cache_dir is not working
)


def ac_infer_batch(fpaths: List[str]):
    """
    Inferencing audio classification model on a batch of audio files.
    With GPU lock to prevent multiple processes from using the GPU at the same time.
    """
    with gpu_lock:
        predictions = pipe(fpaths)
        return predictions


def ac_get_speech_probs(predictions):
    """
    Get the probability of speech in the audio.
    Ajdust the score based on the label.
    """
    speech_probs = []
    for pred in predictions:
        score = 0
        for item in pred:
            # Positive score
            if (
                item["label"] == "Narration, monologue"
                or item["label"] == "Female speech, woman speaking"
                or item["label"] == "Male speech, man speaking"
                or item["label"] == "Speech"
            ):
                score += item["score"]
            # Negative score
            if (
                item["label"] == "Conversation"
                or item["label"] == "Music"
                or item["label"] == "Sound effect"
            ):
                score -= item["score"]
            # Else
        speech_probs.append(score)
    return speech_probs


if __name__ == "__main__":
    def test_ac_infer_batch():
        results = ac_infer_batch(
            [
                "data/audio_sample/clean/HOHDaZ3BLxY_00000074.wav",
                "data/audio_sample/clean/p232_258.wav",
            ]
        )
        print(results)


    def test_ac_get_speech_probs():
        predictions = [[{"label": "Speech", "score": 0.9}, {"label": "Music", "score": 0.1}]]
        speech_probs = ac_get_speech_probs(predictions)
        print(speech_probs)


    from fire import Fire
    Fire()
