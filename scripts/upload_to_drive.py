from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# GANTI dengan folder ID Drive kamu
FOLDER_ID = "1ycZSe3JOL0avT2peCq8ubQoZB97Mwox4"
ARTIFACTS_DIR = "model_output"

gauth = GoogleAuth()
gauth.LoadCredentialsFile("./client_secrets.json")
if not gauth.credentials:
    gauth.LocalWebserverAuth()
else:
    gauth.Authorize()
drive = GoogleDrive(gauth)

for root, dirs, files in os.walk(ARTIFACTS_DIR):
    for filename in files:
        file_path = os.path.join(root, filename)
        print(f"Uploading {file_path} to GDrive folder ...")
        gfile = drive.CreateFile(
            {'title': filename, 'parents': [{'id': FOLDER_ID}]})
        gfile.SetContentFile(file_path)
        gfile.Upload()
        print(f"Uploaded {filename}")
