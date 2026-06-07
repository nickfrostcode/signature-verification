import requests
import os

# Resolve paths to the images in the root repository regardless of where the script is run
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(API_DIR, '..', '..'))

IMG_001 = os.path.join(ROOT_DIR, "001.jpg")
IMG_002 = os.path.join(ROOT_DIR, "002.jpg")

def test_baseline():
    print("Testing baseline...")
    url = "http://localhost:8000/verify/baseline"
    files = {"image": ("001.jpg", open(IMG_001, "rb"), "image/jpeg")}
    data = {"explain": True}
    response = requests.post(url, files=files, data=data)
    print(response.status_code)
    print(response.json())

def test_siamese():
    print("Testing siamese...")
    url = "http://localhost:8000/verify/siamese"
    files = {
        "image_a": ("001.jpg", open(IMG_001, "rb"), "image/jpeg"),
        "image_b": ("002.jpg", open(IMG_002, "rb"), "image/jpeg")
    }
    data = {"explain": True}
    response = requests.post(url, files=files, data=data)
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    test_baseline()
    test_siamese()
