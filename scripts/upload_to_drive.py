from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

SERVICE_ACCOUNT_FILE = 'service_account.json'
FOLDER_ID = "1ycZSe3JOL0avT2peCq8ubQoZB97Mwox4"
ARTIFACTS_DIR = "model_output"

SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

for root, dirs, files in os.walk(ARTIFACTS_DIR):
    for filename in files:
        file_path = os.path.join(root, filename)
        file_metadata = {'name': filename, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()
        print(f"Uploaded {filename} with ID: {file.get('id')}")
