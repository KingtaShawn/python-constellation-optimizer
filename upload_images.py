import base64
import os
from github import Github

# GitHub repository information
REPO_OWNER = "KingtaShawn"
REPO_NAME = "python-constellation-optimizer"
BRANCH = "main"

# Initialize GitHub connection (requires GITHUB_TOKEN environment variable)
g = Github(os.environ.get("GITHUB_TOKEN"))
repo = g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")

# List of important visualization files to upload
image_files = [
    "constellation_comparison.png",
    "ber_comparison.png",
    "training_loss.png"
]

def upload_image_as_base64(file_path):
    """Upload an image file to GitHub as base64 encoded content"""
    try:
        # Read the image file and encode it in base64
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get the filename
        filename = os.path.basename(file_path)
        
        # Try to get the file (to see if it exists)
        try:
            contents = repo.get_contents(filename, ref=BRANCH)
            # If file exists, update it
            repo.update_file(
                path=filename,
                message=f"Update visualization: {filename}",
                content=encoded_string,
                sha=contents.sha,
                branch=BRANCH
            )
            print(f"Updated {filename} in repository")
        except:
            # If file doesn't exist, create it
            repo.create_file(
                path=filename,
                message=f"Add visualization: {filename}",
                content=encoded_string,
                branch=BRANCH
            )
            print(f"Created {filename} in repository")
            
    except Exception as e:
        print(f"Error uploading {file_path}: {str(e)}")

if __name__ == "__main__":
    # Upload each image file
    for image_file in image_files:
        if os.path.exists(image_file):
            upload_image_as_base64(image_file)
        else:
            print(f"File {image_file} not found")