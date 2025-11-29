import requests

url = "http://127.0.0.1:5000/run_inference"
files = {
    "audio_file": open("model/Nee-kavithaigala.wav", "rb"),
    "model_file": open("model/bala_sir2.pth", "rb"),
    "index_file": open("model/bala_sir.index", "rb"),
}
data = {
    "pitch_shift": "0",
    "index_rate": "0.3",
    "volume_envelope": "1.0",
    "protect": "0.33",
    "f0_method": "rmvpe"
}

response = requests.post(url, files=files, data=data)
print(response.json())