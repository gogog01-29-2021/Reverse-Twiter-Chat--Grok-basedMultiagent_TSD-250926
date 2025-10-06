import time
import hmac
import hashlib
import base64
import json
from urllib.parse import urlencode

def get_binance_headers(api_key: str, secret: str, payload: dict):
    timestamp = str(int(time.time() * 1000))
    payload["timestamp"] = timestamp

    query_string = urlencode(payload)

    signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

    return {
        "X-MBX-APIKEY": api_key
    }, f"{query_string}&signature={signature}"
    
def get_bybit_headers(api_key: str, secret: str, payload: dict):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    payload_str = json.dumps(payload, separators=(",", ":"))
    signature_payload = f"{timestamp}{api_key}{recv_window}{payload_str}"
    signature = hmac.new(secret.encode(), signature_payload.encode(), hashlib.sha256).hexdigest()

    return {
        "X-BYBIT-API-KEY": api_key,
        "X-BYBIT-SIGN": signature,
        "X-BYBIT-TIMESTAMP": timestamp,
        "X-BYBIT-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
def get_coinbase_headers(api_key: str, secret: str, passphrase: str, method: str, request_path: str, body: str):
    timestamp = str(time.time())
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    clean_secret = secret.strip().replace("\n", "").replace(" ", "")
    missing_padding = len(clean_secret) % 4
    if missing_padding:
        clean_secret += "=" * (4 - missing_padding)

    secret_bytes = base64.b64decode(clean_secret)
    signature = hmac.new(secret_bytes, message.encode(), hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()

    return {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-SIGN": signature_b64,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json"
    }
    


def get_okx_headers(api_key: str, secret_key: str, passphrase: str, method: str, request_path: str, body: str):
    timestamp = str(time.time())
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    signature = base64.b64encode(
        hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
    ).decode()

    return {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json"
    }
    
def validate_credentials(keys: dict, required_keys: list[str]):
    missing = [k for k in required_keys if k not in keys or not keys[k]]
    if missing:
        raise ValueError(f"Missing credentials: {', '.join(missing)}")