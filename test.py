# # %%writefile test.py
# import logging

# # Set up logging to print to console
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Test logging
# logging.info("Info")
# logging.debug("Debug")


import yt_dlp

def get_video_duration(url):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get('duration', None)  # Duration in seconds
        return duration

# Example usage
url = "https://www.youtube.com/watch?v=68iuJyFMOEA"

def test_get_video_duration():
    duration = get_video_duration(url)
    assert duration is not None, "Could not retrieve duration"
    print(f"Duration: {duration} seconds")


def test_check_audio_quality_48k():
    from yt_download import check_audio_quality_48k
    for i in range(1000):
        rs = check_audio_quality_48k(url)
        print(rs)

if __name__ == "__main__":
    from fire import Fire
    Fire()