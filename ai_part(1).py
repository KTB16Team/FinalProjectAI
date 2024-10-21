from transformers import pipeline
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 감정 분석 모델
emotion_classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# 사용자 프로필 (각 사용자마다 민감도를 설정) (일단 이건 사용자가 많아지면 가능할지는 의문, 일단 예시로)
user_profiles = {
    "userA": {
        "coffee_with_opposite_gender": 20,  # 이성 친구와 커피에 대한 민감도 (기본: 10)
        "alcohol_with_opposite_gender": 40  # 이성 친구와 술에 대한 민감도 (기본: 30)
    },
    "userB": {
        "coffee_with_opposite_gender": 10,  # 이성 친구와 커피는 비교적 낮은 민감도
        "alcohol_with_opposite_gender": 50  # 술에 대한 높은 민감도
    }
}

# 사용자별 맞춤형 가중치 결정 함수
def determine_personalized_weight(speech, user_id):
    # 사용자의 프로필 불러오기
    user_profile = user_profiles.get(user_id, {})
    
    # 기본 민감도 설정
    coffee_weight = user_profile.get("coffee_with_opposite_gender", 10)
    alcohol_weight = user_profile.get("alcohol_with_opposite_gender", 30)
    
    # 가중치 계산 (근데 이게 매번 대화 내용 내에서 캐치하기가,,,,문제일듯 경우의 수 너무 많아지지 않을까하는 의문)
    if "커피" in speech and "이성친구" in speech:
        return coffee_weight
    elif "술" in speech and "이성친구" in speech:
        return alcohol_weight
    else:
        return 10  # 기본 가중치

# 가변적 가중치 설정을 위한 예측 모델 사용
def predict_weight_dynamic(speech, user_id):
    # 감정 분석을 통한 감정 점수 예측
    sentiment_result = emotion_classifier(speech)[0]
    sentiment_score = int(sentiment_result['label'].split()[0])  # 1 ~ 5 범위 감정 점수
    
    # 사용자 맞춤형 가중치 결정 (동적으로 최대한)
    personalized_weight = determine_personalized_weight(speech, user_id)
    
    # 최종 예측된 가중치 조정
    final_weight = sentiment_score * personalized_weight / 10
    return max(0, final_weight)  # 가중치가 음수가 되지 않도록 보정

# 행동 평가 및 가중치 예측 결합 (개인화된 동적 평가)
def evaluate_behavior_automatically(speech, person_type, user_id):
    behavior_score = 0
    explanation = []

    # 예측된 가중치 사용
    weight = predict_weight_dynamic(speech, user_id)

    # 행동 평가의 질문 예시 (유동적으로 상황을 평가)
    explanation.append(f"{person_type}의 행동이 분석되었습니다. 개인화된 가중치는 {weight:.2f}점입니다.")
    return weight, explanation

# 최종 과실 비율 및 보고서 작성
def generate_report(conversation, user_id):
    scores = {}
    explanations = {}
    
    for person, speech in conversation.items():
        # 행동 평가 (개인화된 가중치 예측)
        behavior_score, explanation = evaluate_behavior_automatically(speech, person, user_id)
        
        scores[person] = behavior_score
        explanations[person] = explanation
    
    # 보고서에 나올 비율 계산
    total_score = sum(scores.values())
    
    if total_score == 0:
        return "과실 없음: 0:0", "양쪽 모두 크게 잘못이 없으니, 서로에게 조금 더 이해와 배려를 부탁드려요. 대화를 통해 서로의 입장을 조금 더 잘 이해할 수 있을 거예요."
    
    남자친구_ratio = (scores["남자친구"] / total_score) * 100
    여자친구_ratio = (scores["여자친구"] / total_score) * 100
    
    # 결과보고서 형식(수정필요)
    report = f"남자친구: {round(남자친구_ratio, 1)}%, 여자친구: {round(여자친구_ratio, 1)}%\n"
    report += "\n<상세 분석>\n"
    report += "남자친구:\n"
    report += "\n".join(explanations["남자친구"]) + "\n"
    report += "여자친구:\n"
    report += "\n".join(explanations["여자친구"]) + "\n"

    return f"과실 비율: 남자친구 {round(남자친구_ratio, 1)}%, 여자친구 {round(여자친구_ratio, 1)}%", report


def get_conversation_from_user():
    print("남자친구와 여자친구의 대화를 입력해주세요.")
    boyfriend_speech = input("남자친구: ")
    girlfriend_speech = input("여자친구: ")
    return {
        "남자친구": boyfriend_speech,
        "여자친구": girlfriend_speech
    }

def main():
    user_id = input("사용자 ID를 입력해주세요 (예: userA, userB): ")
    conversation = get_conversation_from_user()
    result, report = generate_report(conversation, user_id)
    
    print("\n<중재 결과>")
    print(result)
    print("\n<상세 보고서>")
    print(report)

if __name__ == "__main__":
    main()