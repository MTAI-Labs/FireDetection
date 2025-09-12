import requests

files = [
    ('files', open('app/local/Images/image1.png', 'rb')),
    ('files', open('app/local/Images/image2.png', 'rb')),
    ('files', open('app/local/Images/image3.jpeg', 'rb'))
]
response = requests.post("http://localhost:8000/api/v1/fire-detection/detect_fire/", files=files)
print(response.json())