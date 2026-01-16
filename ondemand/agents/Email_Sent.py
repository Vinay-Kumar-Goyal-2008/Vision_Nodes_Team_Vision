import json
import os
import sys
import uuid
import requests
from typing import List, Dict, Optional
from flask import Blueprint,jsonify,request
import json

API_KEY = "2TYJCsiaY3x86YP9Tb7ZbGg25u8mPfB2"
BASE_URL = "https://api.on-demand.io/chat/v1"
MEDIA_BASE_URL = "https://api.on-demand.io/media/v1"

EXTERNAL_USER_ID = "<your_external_user_id>"
QUERY = "<your_query>"
RESPONSE_MODE = "stream"  # Now dynamic
AGENT_IDS = ["agent-1712327325","agent-1713962163"]  # Dynamic list from PluginIds
FILE_AGENT_IDS = ["agent-1713954536","agent-1713958591","agent-1713958830","agent-1713961903","agent-1713967141"]
ENDPOINT_ID = "predefined-xai-grok4.1-fast"
REASONING_MODE = "grok-4-fast"
FULFILLMENT_PROMPT = "The Email Communication Agent is designed to send emails to specified email addresses with customizable subjects and message content. It facilitates reliable and structured email communication by focusing strictly on delivering the provided content without modification or interpretation. The agent ensures clarity and accuracy in email delivery, supports standard email formatting, and avoids adding personal opinions or unnecessary text, making it suitable for automated notifications, alerts, and general communication purposes."
STOP_SEQUENCES = []  # Dynamic list
TEMPERATURE = 0.7
TOP_P = 1
MAX_TOKENS = 0
PRESENCE_PENALTY = 0
FREQUENCY_PENALTY = 0

# File upload configuration
FILE_PATH = "<path_to_your_file>"  # e.g., "/Users/username/Downloads/image.png"
FILE_NAME = "<file_name>"  # e.g., "image.png"
CREATED_BY = "AIREV"
UPDATED_BY = "AIREV"

class ContextField:
    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

class SessionData:
    def __init__(self, id: str, context_metadata: List[ContextField]):
        self.id = id
        self.context_metadata = context_metadata

class CreateSessionResponse:
    def __init__(self, data: SessionData):
        self.data = data

class MediaUploadResponse:
    def __init__(self, data: Dict):
        self.data = data

def upload_media_file(file_path: str, file_name: str, agents: List[str], session_id: str) -> Optional[Dict]:
    """
    Upload a media file to the API

    Args:
        file_path: Path to the file to upload
        file_name: Name for the uploaded file
        plugins: List of plugin IDs to process the file

    Returns:
        Dictionary containing upload response data or None if failed
    """
    url = f"{MEDIA_BASE_URL}/public/file/raw"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    print(f"\n--- Uploading Media File ---")
    print(f"📁 File: {file_path}")
    print(f"📝 Name: {file_name}")
    print(f"🔌 Plugins: {plugins}")

    headers = {
        "apikey": API_KEY
    }

    # Prepare multipart form data
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'))
    }

    data = {
        'createdBy': CREATED_BY,
        'updatedBy': UPDATED_BY,
        'name': file_name,
        'responseMode': RESPONSE_MODE,
        "sessionId": session_id,
    }

    # Add plugins to the form data
    for agent in agents:
        if 'agents' not in data:
            data['agents'] = []
        data['agents'].append(agent)

    try:
        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 201 or response.status_code == 200:
            media_response = response.json()
            print(f"✅ Media file uploaded successfully!")
            print(f"📄 File ID: {media_response['data']['id']}")
            print(f"🔗 URL: {media_response['data']['url']}")

            if 'context' in media_response['data']:
                print(f"📋 Context: {media_response['data']['context'][:200]}...")

            return media_response['data']
        else:
            print(f"❌ Error uploading media file: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception during file upload: {str(e)}")
        return None
    finally:
        files['file'][1].close()

def main(query):
    if API_KEY == "<your_api_key>" or not API_KEY:
        print("❌ Please set API_KEY.")
        sys.exit(1)

    global EXTERNAL_USER_ID
    if EXTERNAL_USER_ID == "<your_external_user_id>" or not EXTERNAL_USER_ID:
        EXTERNAL_USER_ID = str(uuid.uuid4())
        print(f"⚠️  Generated EXTERNAL_USER_ID: {EXTERNAL_USER_ID}")
    
    context_metadata = [
        {"key": "userId", "value": "1"},
        {"key": "name", "value": "John"},
    ]


    session_id = create_chat_session(context_metadata)
    if session_id:
        print("\n--- Submitting Query ---")
        print(f"Using query: '{query}'")
        print(f"Using responseMode: '{RESPONSE_MODE}'")
        # Optional: Upload media file if configured
        media_data = None
        if FILE_PATH != "<path_to_your_file>" and FILE_PATH and os.path.exists(FILE_PATH):
            media_data = upload_media_file(FILE_PATH, FILE_NAME, FILE_AGENT_IDS, session_id)
            if media_data:
                print(f"\n✅ Media uploaded. You can reference it in your query or session.")
        return submit_query(session_id, context_metadata)
        

def create_chat_session(context_metadata: List[Dict[str, str]]) -> str:
    url = BASE_URL + "/sessions"
    print(EXTERNAL_USER_ID)
    body = {
        "agentIds": AGENT_IDS,
        "externalUserId": EXTERNAL_USER_ID,
        "contextMetadata": context_metadata,
    }

    json_body = json.dumps(body)

    print(f"\n--- Creating Chat Session ---")
    print(f"📡 Creating session with URL: {url}")
    print(f"📝 Request body: {json_body}")

    headers = {
        "apikey": API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json_body, headers=headers)

    if response.status_code == 201:
        session_resp_data = response.json()
        session_data = SessionData(
            id=session_resp_data["data"]["id"],
            context_metadata=[ContextField(field["key"], field["value"]) for field in session_resp_data["data"].get("contextMetadata", [])]
        )
        session_resp = CreateSessionResponse(data=session_data)

        print(f"✅ Chat session created. Session ID: {session_resp.data.id}")

        if session_resp.data.context_metadata:
            print("📋 Context Metadata:")
            for field in session_resp.data.context_metadata:
                print(f" - {field.key}: {field.value}")

        return session_resp.data.id
    else:
        print(f"❌ Error creating chat session: {response.status_code} - {response.text}")
        return ""

def submit_query(session_id: str, context_metadata: List[Dict[str, str]]):
    url = f"{BASE_URL}/sessions/{session_id}/query"
    body = {
        "endpointId": ENDPOINT_ID,
        "query": QUERY,
        "agentIds": AGENT_IDS,
        "responseMode": RESPONSE_MODE,
        "reasoningMode": REASONING_MODE,
        "modelConfigs": {
            "fulfillmentPrompt": FULFILLMENT_PROMPT,
            "stopSequences": STOP_SEQUENCES,
            "temperature": TEMPERATURE,
            "topP": TOP_P,
            "maxTokens": MAX_TOKENS,
            "presencePenalty": PRESENCE_PENALTY,
            "frequencyPenalty": FREQUENCY_PENALTY,
        },
    }

    json_body = json.dumps(body)

    print(f"🚀 Submitting query to URL: {url}")
    print(f"📝 Request body: {json_body}")

    headers = {
        "apikey": API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json_body, headers=headers, stream=RESPONSE_MODE == "stream")

    print()

    if RESPONSE_MODE == "sync":
        if response.status_code == 201:
            original = response.json()

            # Append context metadata at the end
            if "data" in original:
                original["data"]["contextMetadata"] = context_metadata

            final = json.dumps(original, indent=2)
            print("✅ Final Response (with contextMetadata appended):")
            print(final)
        else:
            print(f"❌ Error submitting sync query: {response.status_code} - {response.text}")
    elif RESPONSE_MODE == "stream":
        print("✅ Streaming Response...")

        full_answer = ""
        final_session_id = ""
        final_message_id = ""
        metrics = {}

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8').strip()

                if line_str.startswith("data:"):
                    data_str = line_str[len("data:"):].strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event.get("eventType") == "fulfillment":
                        if "answer" in event:
                            full_answer += event["answer"]
                        if "sessionId" in event:
                            final_session_id = event["sessionId"]
                        if "messageId" in event:
                            final_message_id = event["messageId"]
                    elif event.get("eventType") == "metricsLog":
                        if "publicMetrics" in event:
                            metrics = event["publicMetrics"]

        final_response = {
            "message": "Chat query submitted successfully",
            "data": {
                "sessionId": final_session_id,
                "messageId": final_message_id,
                "answer": full_answer,
                "metrics": metrics,
                "status": "completed",
                "contextMetadata": context_metadata,
            },
        }

        formatted = json.dumps(final_response, indent=2)
        print("\n✅ Final Response (with contextMetadata appended):")
        return formatted

email = Blueprint('email', __name__)

@email.route("/email_send", methods=["POST"])
def email_send():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    links = data.get("links")  # Expecting: {"links": ["url1", "url2"]}

    if not links or not isinstance(links, list):
        return jsonify({"error": "Provide a list of links"}), 400

    # Build the query prompt
    prompt = f"Act as a web content extractor. Fetch content from provided links: {links}, clean and sanitize the text, and return the extracted information. Handle multiple links in parallel and manage errors gracefully to ensure accurate and complete results."

    # Call your main function with this query
    result = main(query=prompt)  
    return jsonify({"result": json.loads(result)['data']['answer']})
if __name__ == "__main__":
    main()