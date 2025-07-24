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

def test_direct_fastapi_uploads(folder_path, upload_url):
    total = 0
    upload_success = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            total += 1
            file_path = os.path.join(root, file)
            content_type = get_content_type(file)
            print("-"*60)
            print(f"Testing {file_path}")
            with open(file_path, 'rb') as f:
                files_data = {'file': (file, f, content_type)}
                response = requests.post(upload_url, files=files_data)
                print(f"  Upload Status: {response.status_code}")
                try:
                    resp_json = response.json()
                    print(f"  Upload Response: {resp_json}")
                    if file.lower().endswith('.docx'):
                        if resp_json.get('converted_pdf'):
                            print(f"  ✔ DOCX to PDF conversion successful: {resp_json['converted_pdf']}")
                        else:
                            print(f"  ✖ DOCX to PDF conversion missing in response!")
                    document_id = resp_json.get('document_id')
                    if response.status_code == 200 and document_id:
                        upload_success += 1
                    else:
                        print("  No document_id returned from upload response.")
                except Exception:
                    print(f"  Upload Response (raw): {response.text}")
    print("\nSummary:")
    print(f"  Total files tested: {total}")
    print(f"  Uploads succeeded: {upload_success}")

if __name__ == "__main__":
    uploads_folder = os.path.join('documents', 'uploads')
    upload_url = 'http://localhost:8000/api/documents/upload'
    test_direct_fastapi_uploads(uploads_folder, upload_url)
