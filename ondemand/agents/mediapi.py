import os
import requests
from flask import Blueprint, jsonify, request

API_KEY = "2TYJCsiaY3x86YP9Tb7ZbGg25u8mPfB2"
MEDIA_BASE_URL = "https://api.on-demand.io/media/v1"
RESPONSE_MODE = "sync"
CREATED_BY = "AIREV"
UPDATED_BY = "AIREV"

FILE_AGENT_IDS = ["agent-1713954536","agent-1713958591"]

# Flask Blueprint for media API
media = Blueprint("media", __name__)

def upload_media_file(file_path: str, file_name: str, agents: list) -> dict:
    """
    Upload a single media file to OnDemand Media API
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    url = f"{MEDIA_BASE_URL}/public/file/raw"
    headers = {"apikey": API_KEY}
    files = {"file": (os.path.basename(file_path), open(file_path, "rb"))}
    data = {
        "createdBy": CREATED_BY,
        "updatedBy": UPDATED_BY,
        "name": file_name,
        "responseMode": RESPONSE_MODE,
        "agents": agents
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code in [200, 201]:
            media_response = response.json()
            return {
                "success": True,
                "fileId": media_response["data"]["id"],
                "url": media_response["data"]["url"]
            }
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        files['file'][1].close()


@media.route("/upload_media", methods=["POST"])
def upload_media_endpoint():
    """
    Flask endpoint to upload media via JSON request:
    {
        "filePath": "path/to/file.png",
        "fileName": "file.png",
        "agents": ["agent-12345"]
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    file_path = data.get("filePath")
    file_name = data.get("fileName")
    agents = data.get("agents", FILE_AGENT_IDS)

    if not file_path or not file_name:
        return jsonify({"error": "filePath and fileName are required"}), 400

    result = upload_media_file(file_path, file_name, agents)
    return jsonify(result)


if __name__ == "__main__":
    # Example usage without Flask
    example_file = "./op.png"
    example_name = "op.jpeg"
    response = upload_media_file(example_file, example_name, FILE_AGENT_IDS)
    print(response)
