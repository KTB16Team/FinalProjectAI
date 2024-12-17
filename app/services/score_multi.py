import torch
from transformers import AutoTokenizer, BertTokenizer
from torch.nn import functional as F
from typing import List, Dict
import re

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.BERTbasedcontext import EmotionAnalyzer

### Config 설정 ###
class Config:
    MAX_LENGTH = 256
    EMOTION_MODEL_PATH = "/Users/alice.kim/Desktop/aa/Final/BERTbasedemotion_model.pt"
    BEHAVIOR_MODEL_PATH = "/Users/alice.kim/Desktop/aa/Final/Behavior_classifier.pt"
    NUM_LABELS = 5

### 행동 카테고리 모델 클래스 ###
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

### 카테고리 점수 매핑 ###
def map_category_score(category):
    score_map = {
        "경쟁": 0,
        "회피": 0,
        "타협": 0.5,
        "협력": 1,
        "수용": 1
    }
    return score_map.get(category, 0)

### 문장 감정 점수 분석 클래스 ###
class SentenceEmotionAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device('cpu')
        self.analyzer = EmotionAnalyzer(model_path=model_path)
        self.tokenizer = self.analyzer.tokenizer
    
    def analyze_sentences(self, sentences: List[str]) -> List[float]:
        results = self.analyzer.analyze_conversation(sentences)
        return [result['emotion_score'] for result in results]

### 행동 카테고리 점수 예측 함수 ###
def predict_category_scores(sentences: List[str], model, tokenizer, label_map, device) -> List[float]:
    reverse_label_map = {v: k for k, v in label_map.items()}
    scores = []
    for text in sentences:
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

        predicted_category = reverse_label_map[predicted_index]
        score = map_category_score(predicted_category)
        scores.append(score)
    return scores

### 인물별 점수 계산 및 백분율 출력 ###
def analyze_and_calculate_percentage(conversation: List[str]):
    # 인물명 제거 및 문장 분리
    pattern = r"^[A-Z]:\s*"  # 인물명 패턴
    speakers = []
    sentences = []

    for line in conversation:
        match = re.match(pattern, line)
        speaker = match.group(0).strip(": ") if match else "Unknown"
        sentence = re.sub(pattern, "", line).strip()
        if sentence:
            speakers.append(speaker)
            sentences.append(sentence)

    # 감정 점수 모델 로드
    emotion_analyzer = SentenceEmotionAnalyzer(model_path=Config.EMOTION_MODEL_PATH)
    emotion_scores = emotion_analyzer.analyze_sentences(sentences)

    # 행동 카테고리 모델 로드
    device = torch.device("cpu")
    behavior_model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)
    behavior_model.load_state_dict(torch.load(Config.BEHAVIOR_MODEL_PATH, map_location=device))
    behavior_model.to(device)
    behavior_model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    label_map = {"경쟁": 0, "회피": 1, "타협": 2, "협력": 3, "수용": 4}

    behavior_scores = predict_category_scores(sentences, behavior_model, tokenizer, label_map, device)

    # 인물별 점수 계산
    speaker_scores = {}
    for idx, speaker in enumerate(speakers):
        product = emotion_scores[idx] * behavior_scores[idx]
        speaker_scores[speaker] = speaker_scores.get(speaker, 0) + product

    # 전체 점수 합산
    total_score = sum(speaker_scores.values())

    # 결과 출력
    print("\n인물별 최종 점수 (백분율):")
    for speaker, score in speaker_scores.items():
        percentage = (score / total_score) * 100 if total_score > 0 else 0
        print(f"{speaker}: {score:.6f} ({percentage:.2f}%)")

### 메인 함수 ###
def main():
    conversation = [
        "A: 너 왜 이렇게 늦었어? 나 30분이나 기다렸잖아.",
        "B: 미안, 차가 너무 막혔어. 나도 빨리 오려고 했는데...",
        "A: 핑계 대지 말고. 약속 시간 좀 지키자.",
        "B: 정말 미안해. 커피라도 내가 살게.",
        "A: 됐어, 커피 하나로 넘어갈 일이 아니야.",
        "B: 화 풀어, 내가 일부러 그런 것도 아니잖아.",
        "A: 그래도 넌 항상 이런 식이야. 나만 바보 되는 기분이야.",
        "B: 그건 좀 너무하네. 나도 바쁘고 힘들 때가 있다고.",
        "A: 알겠어. 서로 이해하려고 노력하자. 이러다 다투기만 하네.",
        "B: 응, 나도 잘할게. 우리 화해하자.",
        "A: 그래, 나도 미안. 너무 예민했어."
    ]

    analyze_and_calculate_percentage(conversation)

if __name__ == "__main__":
    main()
