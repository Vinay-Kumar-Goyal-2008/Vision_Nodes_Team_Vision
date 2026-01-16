#   # Now dynamic
# AGENT_IDS = ["agent-1712327325","agent-1713962163"]
# ENDPOINT_ID = "predefined-xai-grok4.1-fast"
# REASONING_MODE = "grok-4-fast"
# FULFILLMENT_PROMPT = "You are an Educational Content Agent."

from .base import BaseAgent

class EducationAgent(BaseAgent):
    agent_ids = ["agent-1712327325", "agent-1713962163"]
    fulfillment_prompt = "You are an Educational Content Agent."

