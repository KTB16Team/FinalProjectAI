import torch
from transformers import BertTokenizer, BertModel
import sys
import os

PROJECT_ROOT = "app"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.behavior_classification import CustomBERTClassifier, Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    
    try:
        model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)
        model_path = os.path.join(PROJECT_ROOT, "services/Behavior_classifier.pt")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        
        return True
    except Exception as e:
        print(f"모델 초기화 실패: {str(e)}")
        return False

def behavior_classification_test(test_text):
    if model is None or tokenizer is None:
        if not init_model():
            return {"error": "모델 초기화 실패"}

    try:
        label_map = {
            0: "경쟁형",
            1: "회피형",
            2: "수용형",
            3: "타협형",
            4: "협력형"
        }
        
        encoded = tokenizer(
            test_text,
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
        
        return {
            "success": True,
            "text": test_text,
            "behavior_type": label_map[prediction],
            "confidence": float(confidence),
            "confidence_level": "높음" if confidence >= 0.8 else "중간" if confidence > 0.4 else "낮음"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if init_model():
        test_text = "프로젝트 진행 상황이 많이 늦어지고 있어요. 이대로 가다가는 기한 내에 끝내기 힘들 것 같은데, 어떻게 생각하세요?"
        result = behavior_classification_test(test_text)
        print("\n테스트 결과:", result)