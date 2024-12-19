import torch
from transformers import AutoTokenizer
from typing import List, Dict

# 모델 클래스와 정의는 제공된 코드에 포함되어 있다고 가정합니다.
# 이미 학습된 모델이 `emotion_model.pt` 파일에 저장되어 있다고 가정합니다.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.BERTbasedcontext import EmotionAnalyzer

class SentenceEmotionAnalyzer:
    def __init__(self, model_path: str = "/Users/alice.kim/Desktop/aa/Final/BERTbasedemotion_model.pt"):
        self.device = torch.device('cpu')
        self.analyzer = EmotionAnalyzer(model_path=model_path)
        self.tokenizer = self.analyzer.tokenizer
    
    def analyze_sentences(self, sentences: List[str]) -> None:
        results = self.analyzer.analyze_conversation(sentences)
        
        for result in results:
            print(f"문장: {result['text']}")
            print(f"점수: {result['emotion_score']:.3f}")

# 텍스트를 입력하면 점수를 계산하는 함수
def main():
    sentences = [
        "그래 내가 잘못한 거 맞아. 근데 너도 잘못한게 없지는 않잖아.",
        "이제 그만하자. 내가 다 잘못했어.",
        "우린 협력해서 문제를 해결해야 해.",
        "난 그냥 피하고 싶어. 그게 편해."
    ]
    
    # 모델 로드 및 분석
    analyzer = SentenceEmotionAnalyzer(model_path="/Users/alice.kim/Desktop/aa/Final/BERTbasedemotion_model.pt")
    analyzer.analyze_sentences(sentences)

if __name__ == "__main__":
    main()
