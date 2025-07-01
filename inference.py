import argparse
from transformers import AutoModelForImageClassification
from preprocess import get_image_processor
from PIL import Image
import torch

def predict(image_path, model_path, device):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Load image processor
    image_processor = get_image_processor(model_path)
    
    # Preprocess image
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
    
    # Get label
    label = model.config.id2label[predicted_class_id]
    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_id].item()

    return predicted_class_id, label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model or model checkpoint directory.")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class_id, label, confidence = predict(args.image_path, args.model_path, device)
    
    print(f"Predicted class ID: {class_id}")
    print(f"Predicted label: {label}")
    print(f"Confidence: {confidence:.4f}")
