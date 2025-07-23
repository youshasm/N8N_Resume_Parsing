import os
import requests

def get_content_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return 'application/pdf'
    elif ext == '.docx':
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif ext == '.doc':
        return 'application/msword'
    elif ext in ['.jpeg', '.jpg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    else:
        return 'application/octet-stream'

def test_uploads(folder_path, webhook_url):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            content_type = get_content_type(file)
            with open(file_path, 'rb') as f:
                files_data = {'file': (file, f, content_type)}
                response = requests.post(webhook_url, files=files_data)
                print(f"Testing {file_path} -> Status: {response.status_code}")
                try:
                    print(response.json())
                except Exception:
                    print(response.text)

if __name__ == "__main__":
    uploads_folder = os.path.join('documents', 'uploads')
    webhook = 'http://localhost:5678/webhook/primary'
    test_uploads(uploads_folder, webhook)
