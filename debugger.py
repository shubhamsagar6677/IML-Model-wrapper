import requests

url = "http://127.0.0.1:6000/api/segment-image"
data = {
    "pixelData": [0] * (512 * 512)  # Example with all zeros
}
response = requests.post(url, json=data)
print(response.status_code)
