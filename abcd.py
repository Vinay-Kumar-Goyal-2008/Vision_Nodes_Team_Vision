import requests
response = requests.get("https://api.on-demand.io")
print(response.status_code)
