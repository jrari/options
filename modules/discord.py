
import json, requests
def send_discord_webhook(url: str, content: str="", username: str="Options Bot", embeds=None, files=None):
    embeds = embeds or []
    if files:
        return requests.post(url, data={"payload_json": json.dumps({"content": content, "username": username, "embeds": embeds})}, files=files, timeout=30)
    return requests.post(url, headers={"Content-Type":"application/json"}, data=json.dumps({"content": content, "username": username, "embeds": embeds}), timeout=30)
