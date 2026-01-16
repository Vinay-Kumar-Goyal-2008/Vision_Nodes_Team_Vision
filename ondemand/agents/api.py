from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app)

# In-memory store (for demo)
ITEMS = {}

# 1. Health check
@app.route("/wikipedia", methods=["GET"])
def wikipedia(page):
    prompt=f'Act as an educational assistant. Provide clear, concise, and accurate explanations on subjects like physics, mathematics, literature, and history using verified information from Wikipedia. Include examples, key concepts, and relevant context to make the topic easy to understand. Keep explanations structured and, when useful, add brief summaries or comparisons.'
# 2. Create item
@app.route("/web_summary", methods=["POST"])
def web_summary(page):
    prompt=f'Act as a web content extractor. Fetch content from provided links, clean and sanitize the text, and return the extracted information. Handle multiple links in parallel and manage errors gracefully to ensure accurate and complete results.'

# 3. Get all items
@app.route("/health_chat", methods=["GET"])
def health_chat():
    prompt=f'Act as a health assistant using reliable sources. Answer any health-related questions clearly, accurately, and with actionable information. Provide context and explanations where needed'

# 4. Get single item
@app.route("/google_books", methods=["GET"])
def google_books():
    prompt=f'Provide book information from a large database. Search by title, author, ISBN, or other criteria, and return metadata including title, author, description, and cover image.'

# 5. Update item
@app.route("/email_sent", methods=["PUT"])
def email_sent(data):
    prompt=f'Sends emails to specified email addresses with customisable subject and content, facilitating email communications.'

# 6. Delete item
@app.route("/air_quality", methods=["DELETE"])
def air_quality(state):
    prompt=f'Act as an air quality assistant. Provide real-time insights on climate and air quality for a user’s location. Answer questions about current air quality, average climate conditions, and potential impacts on health, delivering clear and actionable information.'

# # 7. Simple auth (mock)
# @app.route("/chatbot", methods=["POST"])
# def login():
#     pass
# @app.route("/mediabot", methods=["GET"])
# def call_external():
#     pass

if __name__ == "__main__":
    app.run(debug=True)