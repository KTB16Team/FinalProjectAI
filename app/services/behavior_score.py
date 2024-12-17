import torch
from transformers import BertTokenizer
from torch.nn import functional as F

class Config:
    MAX_LENGTH = 256
    SAVE_PATH = "Behavior_classifier.pt"  # 모델 저장 경로
    NUM_LABELS = 5

class CustomBERTClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def map_category_score(category):
    """카테고리별 점수를 매핑"""
    score_map = {
        "경쟁": 0,
        "회피": 0,
        "타협": 0.5,
        "협력": 1,
        "수용": 1
    }
    return score_map.get(category, 0)

def load_model_and_tokenizer(model_path, device):
    """모델과 토크나이저 불러오기"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_category_score(texts, model, tokenizer, label_map, device):
    """텍스트를 입력받아 점수를 반환"""
    reverse_label_map = {v: k for k, v in label_map.items()}
    scores = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = F.softmax(logits, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()

        # 예측된 카테고리와 점수 매핑
        reverse_label_map = {v: k for k, v in label_map.items()}
        predicted_category = reverse_label_map[predicted_index]
        score = map_category_score(predicted_category)
        scores.append((text, score))
    return scores

if __name__ == "__main__":
    # 설정
    model_path = "/Users/alice.kim/Desktop/aa/Final/Behavior_classifier.pt"  # 학습된 모델 경로
    label_map = {"경쟁": 0, "회피": 1, "타협": 2, "협력": 3, "수용": 4}  # 레이블 매핑
    input_texts = [
        "그래 내가 잘못한 거 맞아. 근데 너도 잘못한게 없지는 않잖아.",
        "이제 그만하자. 내가 다 잘못했어.",
        "우린 협력해서 문제를 해결해야 해.",
        "난 그냥 피하고 싶어. 그게 편해.",
    ]

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 및 토크나이저 불러오기
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    scores = predict_category_score(input_texts, model, tokenizer, label_map, device)
    # 점수 예측
    score = predict_category_score(input_texts, model, tokenizer, label_map, device)
    for text, score in scores:
        print(f"문장: {text}\n점수: {score}")
