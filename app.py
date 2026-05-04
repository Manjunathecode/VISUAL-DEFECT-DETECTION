import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Load Model ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(
    'defect_detection_model.pth', map_location=device))
model.eval()
model.to(device)

# ── Image Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Prediction Function ──


def predict(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = {0: "❌ DEFECTIVE", 1: "✅ NON-DEFECTIVE"}
    result = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    return f"{result}\nConfidence: {confidence_score:.2f}%"


# ── Gradio Interface ──
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Product Image"),
    outputs=gr.Textbox(label="Detection Result"),
    title="🔍 AI-Powered Visual Defect Detection System",
    description="Upload a casting product image to detect if it is Defective or Non-Defective.",
    examples=[
        ["casting_512x512/def_front/cast_def_0_0.jpeg"],
        ["casting_512x512/ok_front/cast_ok_0_0.jpeg"]
    ],
    theme=gr.themes.Soft()
)

interface.launch()
