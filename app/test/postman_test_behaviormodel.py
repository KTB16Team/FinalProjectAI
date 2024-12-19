from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer
import torch
import uvicorn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.services.behavior_classification import CustomBERTClassifier, Config

app = FastAPI(title="Behavior Classification API")

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    try:
        model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)
        model_path = os.path.join("app", "services", "/Users/alice.kim/Desktop/aa/Final/Behavior_classifier.pt")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return True
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        return False

if not init_model():
    raise RuntimeError("Model initialization failed!")

# Request and Response Models
class BehaviorClassificationRequest(BaseModel):
    text: str

class BehaviorClassificationResponse(BaseModel):
    success: bool
    behavior_type: str
    confidence: float
    confidence_level: str

# Behavior Classification Endpoint
@app.post("/api/v1/private-posts/classify", response_model=BehaviorClassificationResponse)
def classify_behavior(request: BehaviorClassificationRequest):
    try:
        label_map = {
            0: "경쟁형",
            1: "회피형",
            2: "수용형",
            3: "타협형",
            4: "협력형"
        }

        encoded = tokenizer(
            request.text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()

        return BehaviorClassificationResponse(
            success=True,
            behavior_type=label_map[prediction],
            confidence=confidence,
            confidence_level="높음" if confidence >= 0.8 else "중간" if confidence > 0.4 else "낮음"
        )
    except Exception as e:
        return BehaviorClassificationResponse(
            success=False,
            behavior_type="",
            confidence=0.0,
            confidence_level=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
