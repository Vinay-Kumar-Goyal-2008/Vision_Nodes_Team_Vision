from flask import Flask
from Air_Quality import air
from Email_Sent import email
from Google_books import google
from Health_chat import health
from wikipedia import wiki
from web_summary import web

app=Flask(__name__)

app.register_blueprint(web)
app.register_blueprint(air)
app.register_blueprint(email)
app.register_blueprint(google)
app.register_blueprint(health)
app.register_blueprint(wiki)


if __name__ == "__main__":
    app.run(debug=True)