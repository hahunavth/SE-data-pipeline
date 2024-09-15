import requests
import csv

# URL của YouTube Data API v3 để tìm kiếm
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# Các từ khóa liên quan đến tài chính bằng tiếng Việt
QUERY = "tài chính OR đầu tư OR chứng khoán OR ngân hàng"

# Hàm thực hiện tìm kiếm các kênh YouTube với từ khóa liên quan đến tài chính
def search_channels(api_key, query, language='vi', max_results=50):
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'channel',
        'maxResults': max_results,
        'key': api_key
    }
    
    response = requests.get(SEARCH_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        channels = []
        for item in data.get('items', []):
            channel_info = {
                'channel_id': item['snippet']['channelId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'language': language,
                'published_at': item['snippet']['publishedAt']
            }
            channels.append(channel_info)
        return channels
    else:
        print(f"Error: {response.status_code}")
        return []

# Lưu danh sách kênh vào file CSV
def save_to_csv(channels, filename='financial_channels.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['channel_id', 'title', 'description', 'language', 'published_at'])
        writer.writeheader()
        writer.writerows(channels)

# Main program
if __name__ == "__main__":
    channels = search_channels(API_KEY, QUERY)
    if channels:
        save_to_csv(channels)
        print(f"Đã lưu {len(channels)} kênh vào file 'financial_channels.csv'.")
    else:
        print("Không tìm thấy kênh nào.")

