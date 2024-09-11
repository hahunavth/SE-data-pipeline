import librosa
import numpy as np


def estimate_snr(audio, frame_length=2048, hop_length=512, noise_threshold=0.02):
    try:
        frames = librosa.util.frame(
            audio, frame_length=frame_length, hop_length=hop_length
        )
        rms_values = np.sqrt(np.mean(frames**2, axis=0))
        noise_frames = rms_values < noise_threshold
        signal_power = np.mean(rms_values[~noise_frames] ** 2)
        noise_power = np.mean(rms_values[noise_frames] ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
    except Exception as e:
        print(e)
        return 0
    return snr
