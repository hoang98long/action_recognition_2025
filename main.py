from fastapi import FastAPI, File, UploadFile
import torch
import timm
from torchvision import transforms
from PIL import Image
import io
import pickle

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b5', pretrained=False, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("models/action/best_efficientnet_model.pt", map_location=device))
model.to(device)
model.eval()


image_size = 224  # EfficientNet-B5 456
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

app = FastAPI(title="Human Action Recognition API",
              description="Dự đoán hành động từ ảnh với EfficientNet-B5",
              version="1.0",
              root_path="/ges")

@app.post("/")
async def predict_action(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()
    return {
        "predicted_action": pred_class,
        "confidence": round(confidence * 100, 2)
    }
