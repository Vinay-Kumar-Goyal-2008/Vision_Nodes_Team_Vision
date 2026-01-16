import requests
import uuid

API_KEY = "<your_api_key>"
BASE_URL = "https://api.on-demand.io/chat/v1"

HEADERS = {
    "apikey": API_KEY,
    "Content-Type": "application/json"
}

def ask_ondemand(query: str, agent):
    payload = {
        "id": str(uuid.uuid4()),
        "endpointId": agent.endpoint_id,
        "query": query,
        "agentIds": agent.agent_ids,
        "reasoningMode": agent.reasoning_mode,
        "responseMode": "sync",
        "modelConfigs": {
            "fulfillmentPrompt": agent.fulfillment_prompt
        }
    }

    response = requests.post(
        f"{BASE_URL}/completions",
        headers=HEADERS,
        json=payload
    )

    response.raise_for_status()
    return response.json()
