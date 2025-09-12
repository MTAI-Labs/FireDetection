import requests

files = [
    ('files', open('app/local/Images/image1.jpg', 'rb')),
    ('files', open('app/local/Images/image2.jpeg', 'rb')),
    ('files', open('app/local/Images/image3.jpeg', 'rb')),
    ('files', open('app/local/Images/image4.jpg', 'rb')),
    ('files', open('app/local/Images/image5.jpeg', 'rb'))
]
response = requests.post("http://localhost:8000/api/v1/fire-detection/detect_fire/", files=files)
print(response.json())