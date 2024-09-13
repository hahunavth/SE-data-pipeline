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
url = "https://www.youtube.com/watch?v=QEb-9Xy2A70"
duration = get_video_duration(url)

if duration is not None:
    print(f"Duration: {duration} seconds")
else:
    print("Could not retrieve duration")
