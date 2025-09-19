import base64
import os
import requests
import json

# GitHub repository information
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "KingtaShawn"
REPO_NAME = "python-constellation-optimizer"
BRANCH = "main"

# Headers for GitHub API
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def upload_file(file_path, commit_message):
    """Upload a file to GitHub repository"""
    try:
        # Read the file content
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Encode content in base64
        encoded_content = base64.b64encode(content).decode('utf-8')
        
        # Get the filename
        filename = os.path.basename(file_path)
        
        # Check if file already exists
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
        params = {"ref": BRANCH}
        response = requests.get(url, headers=headers, params=params)
        
        data = {
            "message": commit_message,
            "content": encoded_content,
            "branch": BRANCH
        }
        
        if response.status_code == 200:
            # File exists, get the SHA
            existing_file = response.json()
            data["sha"] = existing_file["sha"]
            print(f"Updating existing file: {filename}")
        else:
            print(f"Creating new file: {filename}")
        
        # Upload the file
        response = requests.put(url, headers=headers, data=json.dumps(data))
        
        if response.status_code in [200, 201]:
            print(f"Successfully uploaded {filename}")
        else:
            print(f"Failed to upload {filename}: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error uploading {file_path}: {str(e)}")

if __name__ == "__main__":
    # List of important visualization files to upload
    image_files = [
        "constellation_comparison.png",
        "ber_comparison.png",
        "training_loss.png"
    ]
    
    # Upload each image file
    for image_file in image_files:
        if os.path.exists(image_file):
            upload_file(image_file, f"Add visualization: {image_file}")
        else:
            print(f"File {image_file} not found")