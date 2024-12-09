from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import re

@dataclass
class DialogueEmotionResult:
    text: str
    speaker: str
    final_score: float
    keyword_score: float
    bert_score: float
    interaction_score: float
    detailed_scores: Dict[str, float]

class EmotionAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
        
        # 감정 사전
        self.emotion_dict = {
            'very_positive': {
                'words': ['축하', '대단', '완벽', '최고', '사랑', '감사', '기쁘'],
                'score': 0.9
            },
            'positive': {
                'words': ['좋', '잘', '괜찮', '힘내', '할 수 있', '응원'],
                'score': 0.6
            },
            'supportive': {
                'words': ['이해', '그렇구나', '그래도', '노력', '믿'],
                'score': 0.4
            },
            'negative': {
                'words': ['힘들', '아쉽', '슬프', '불안', '걱정', '안좋'],
                'score': -0.6
            },
            'very_negative': {
                'words': ['최악', '절망', '포기', '실망', '화나', '짜증', '싫'],
                'score': -0.9
            }
        }
        
        # 상호작용 패턴
        self.interaction_patterns = {
            'empathy': {
                'words': ['이해해', '그렇구나', '맞아', '그런 거 같아'],
                'score': 0.6
            },
            'encouragement': {
                'words': ['힘내', '할 수 있', '괜찮아', '좋아질'],
                'score': 0.7
            },
            'concern': {
                'words': ['걱정', '괜찮아', '어떡해', '다행'],
                'score': 0.5
            },
            'agreement': {
                'words': ['그래', '맞아', '당연', '동의'],
                'score': 0.4
            },
            'consolation': {
                'words': ['위로', '힘들었', '수고', '고생'],
                'score': 0.6
            }
        }

    def _analyze_emotion_keywords(self, text: str) -> float:
        base_score = 0
        
        # 기본 감정 단어 분석
        for category, data in self.emotion_dict.items():
            for word in data['words']:
                if word in text:
                    base_score += data['score']
                    
        return np.clip(base_score, -1, 1)

    def _analyze_bert_context(self, text: str) -> float:
        # BERT 기반 문맥 분석
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            bert_output = outputs.last_hidden_state[:, 0].mean().item()
            
        return np.clip(bert_output, -1, 1)

    def _calculate_interaction(self, current_text: str, prev_text: str, prev_emotion: float) -> float:
        if not prev_text:
            return 0.0
        
        interaction_score = 0.0
        
        # 기본 상호작용 패턴 분석
        for pattern_type, data in self.interaction_patterns.items():
            for word in data['words']:
                if word in current_text:
                    interaction_score += data['score']
        
        # 이전 발화가 부정적일 때 위로/격려 보너스
        if prev_emotion < -0.3:
            if any(word in current_text for word in self.emotion_dict['positive']['words']):
                interaction_score += 0.4
            if any(word in current_text for word in self.interaction_patterns['encouragement']['words']):
                interaction_score += 0.3
        
        # 감사 표현 보너스
        gratitude_words = ['고마워', '감사', '든든']
        if any(word in current_text for word in gratitude_words):
            interaction_score += 0.5
            
        return np.clip(interaction_score, -1, 1)

    def calculate_final_score(self, text: str, prev_text: Optional[str] = None, prev_emotion: float = 0.0) -> Dict[str, float]:
        # 1. 키워드 기반 감정 분석 (40%)
        keyword_score = self._analyze_emotion_keywords(text)
        
        # 2. BERT 문맥 분석 (35%)
        bert_score = self._analyze_bert_context(text)
        
        # 3. 상호작용 분석 (25%)
        interaction_score = self._calculate_interaction(text, prev_text, prev_emotion)
        
        # 각 점수의 세부 내용 계산
        detailed_scores = {
            'keyword_base': keyword_score * 0.4,
            'context_base': bert_score * 0.35,
            'interaction_base': interaction_score * 0.25
        }
        
        # 최종 감정 점수 계산
        final_score = sum(detailed_scores.values())
        
        # 추가 조정: 상호작용 가중치 동적 조정
        if abs(interaction_score) > 0.7:  # 강한 상호작용이 있는 경우
            interaction_weight = min(0.35, abs(interaction_score) * 0.25)
            final_score = (
                keyword_score * (0.4 - interaction_weight/2) +
                bert_score * (0.35 - interaction_weight/2) +
                interaction_score * (0.25 + interaction_weight)
            )
        
        # 문맥 증폭 효과
        if abs(bert_score - keyword_score) > 0.5:
            context_weight = 0.1
            final_score = final_score * (1 + context_weight * np.sign(bert_score))
        
        return {
            'final_score': np.clip(final_score, -1, 1),
            'keyword_score': keyword_score,
            'bert_score': bert_score,
            'interaction_score': interaction_score,
            'detailed_scores': detailed_scores
        }

    def analyze_dialogue(self, texts: List[str], speakers: List[str]) -> List[DialogueEmotionResult]:
        results = []
        prev_text = None
        prev_emotion = 0.0
        
        for text, speaker in zip(texts, speakers):
            # 현재 발화 분석
            scores = self.calculate_final_score(text, prev_text, prev_emotion)
            
            # 결과 생성
            result = DialogueEmotionResult(
                text=text,
                speaker=speaker,
                final_score=scores['final_score'],
                keyword_score=scores['keyword_score'],
                bert_score=scores['bert_score'],
                interaction_score=scores['interaction_score'],
                detailed_scores=scores['detailed_scores']
            )
            
            results.append(result)
            
            # 다음 분석을 위한 정보 업데이트
            prev_text = text
            prev_emotion = scores['final_score']
            
        return results

def format_emotion_score(score: float) -> str:
    if score >= 0.6: return "매우 긍정적"
    elif score >= 0.2: return "긍정적"
    elif score > -0.2: return "중립적"
    elif score > -0.6: return "부정적"
    else: return "매우 부정적"

def main():
    analyzer = EmotionAnalyzer()
    
    # 테스트 대화
    conversation = [
        ("팀장님이 오늘 또 내 기획안을 다 엎으래...", "A"),
        ("정말? 이번에도? 너무하시네...", "B"),
        ("여러번 수정했는데도 맘에 안 든대.", "A"),
        ("그동안 엄청 고생했잖아... 많이 힘들었겠다", "B"),
        ("진짜 이직하고 싶은데 지금이 적절한 타이밍일까?", "A"),
        ("그래도 좀 더 버텨보는 건 어때? 네 실력 좋은 거 팀장님도 알걸?", "B"),
        ("모르겠다... 집에 가서 맥주나 마셔야겠어", "A"),
        ("그래! 푹 쉬고! 내일은 더 나아질 거야!", "B")
    ]
    
    texts, speakers = zip(*conversation)
    results = analyzer.analyze_dialogue(list(texts), list(speakers))
    
    print("\n=== 통합 감정 분석 결과 ===")
    for i, result in enumerate(results, 1):
        print(f"\n발화 {i}")
        print(f"화자: {result.speaker}")
        print(f"텍스트: {result.text}")
        print(f"감정 상태: {format_emotion_score(result.final_score)}")
        print(f"최종 감정 점수: {result.final_score:.2f}")
        print("세부 점수:")
        print(f"- 키워드 기반 점수: {result.keyword_score:.2f}")
        print(f"- 문맥 분석 점수: {result.bert_score:.2f}")
        print(f"- 상호작용 점수: {result.interaction_score:.2f}")
        print("가중치 적용 점수:")
        for key, value in result.detailed_scores.items():
            print(f"- {key}: {value:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()