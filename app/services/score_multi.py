import torch
from transformers import AutoTokenizer, BertTokenizer
from torch.nn import functional as F
from typing import List, Dict
from dotenv import load_dotenv
from pathlib import Path
import re
import json
from openai import AsyncOpenAI
import asyncio
import sys
import os

from services.BERTbasedcontext import EmotionAnalyzer

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)



class Config:
    MAX_LENGTH = 256
    EMOTION_MODEL_PATH = "app/services/BERTbasedemotion_model.pt"
    BEHAVIOR_MODEL_PATH = "app/services/Behavior_classifier.pt"
    NUM_LABELS = 5
    GPT_MODEL = "gpt-4"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class SituationAnalyzer:
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    async def analyze_text(self, text: str) -> Dict[str, Dict[str, any]]:
        prompt = f"""Analyze the following text:
        1. Categorize the main conflicts/situations occurring in this context.
        2. Calculate the importance (%) of each situation in the overall conflict.
        3. Extract the relevant sentences for each situation.
        Please print the output in Korean.
        Take a deep breath and solve it step by step.
        Text:
        {text}

        Please respond in the following JSON format:
        {{
            "situations": [
                {{
                    "category": "situation category name",
                    "importance": importance of the situation (a decimal between 0 and 1),
                    "related_sentences": ["related sentence 1", "related sentence 2", ...]
                }},
                ...
            ]
        }}"""

        try:
            response = await self.client.chat.completions.create(
                model=Config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing conflict situations in text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = json.loads(response.choices[0].message.content)
            situation_data = {}
            for situation in analysis["situations"]:
                situation_data[situation["category"]] = {
                    "importance": situation["importance"],
                    "sentences": situation["related_sentences"]
                }
            
            return situation_data
            
        except Exception as e:
            print(f"Error in situation analysis: {e}")
            return {}

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
    score_map = {
        "경쟁": 0,
        "회피": 0,
        "타협": 0.5,
        "협력": 1,
        "수용": 1
    }
    return score_map.get(category, 0)

class SentenceEmotionAnalyzer:
    def __init__(self, model_path: str):
        self.device = torch.device('cpu')
        self.analyzer = EmotionAnalyzer(model_path=model_path)
        self.tokenizer = self.analyzer.tokenizer
    
    def analyze_sentences(self, sentences: List[str]) -> List[float]:
        results = self.analyzer.analyze_conversation(sentences)
        return [result['emotion_score'] for result in results]

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

class ConflictAnalyzer:
    def __init__(self):
        self.emotion_analyzer = SentenceEmotionAnalyzer(Config.EMOTION_MODEL_PATH)
        self.situation_analyzer = SituationAnalyzer()
        self.device = torch.device("cpu")
        self.behavior_model = self._load_behavior_model()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.label_map = {"경쟁": 0, "회피": 1, "타협": 2, "협력": 3, "수용": 4}
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    def _load_behavior_model(self):
        model = CustomBERTClassifier(Config.NUM_LABELS)
        model.load_state_dict(torch.load(Config.BEHAVIOR_MODEL_PATH, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    async def analyze_content(self, content: str, request_id: str = "id") -> dict:
        summary_prompt = f"""Analyze the following content and extract the title, plaintiff's stance, and defendant's stance:
        {content}
        
        Format:
        {{
            "title": "Brief title",
            "stancePlaintiff": "Plaintiff's stance",
            "stanceDefendant": "Defendant's stance",
            "summary": "Overall summary of the situation"
        }}
        Please print the output in Korean.
        """
        
        try:
            summary_response = await self.client.chat.completions.create(
                model=Config.GPT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2
            )
            basic_info = json.loads(summary_response.choices[0].message.content)
        except Exception as e:
            print(f"Error in summary generation: {e}")
            return {"status": "error", "message": "Failed to generate summary"}

        situation_data = await self.situation_analyzer.analyze_text(content)
        
        speaker_scores = {"plaintiff": 0, "defendant": 0}
        total_importance = 0
        
        for situation, data in situation_data.items():
            sentences = data["sentences"]
            importance = data["importance"]
            total_importance += importance

            emotion_scores = self.emotion_analyzer.analyze_sentences(sentences)
            behavior_scores = predict_category_scores(
                sentences, 
                self.behavior_model, 
                self.tokenizer, 
                self.label_map, 
                self.device
            )
            
            for i, (e_score, b_score) in enumerate(zip(emotion_scores, behavior_scores)):
                sentence_score = e_score * b_score * importance
                
                sentence = sentences[i]
                if "내가" in sentence or "나는" in sentence or "나의" in sentence or "내" in sentence:
                    speaker_scores["plaintiff"] += sentence_score
                else:
                    speaker_scores["defendant"] += sentence_score

        total_score = speaker_scores["plaintiff"] + speaker_scores["defendant"]
        
        if total_score > 0:
            fault_rate = round((speaker_scores["plaintiff"] / total_score) * 100,2)
            fault_rate = max(0, min(100, fault_rate))
        else:
            fault_rate = 50.0 
        
        judgment_prompt = f"""You are a warm and empathetic psychological counselor like Dr. Oh Eunyeong. 
        Analyze the following conflict situation, understand the psychology of both parties, and provide advice for healing and reconciliation.

        When presenting, please follow these guidelines:
        1. First, empathize with the client's emotions and let them know that these feelings are natural and valid.
        2. Carefully acknowledge the other party's perspective, but clearly point out their mistakes.
        3. Suggest specific conversation methods or behaviors to resolve the issue.
        4. Conclude with warm words of encouragement.
        Please print the output in Korean.
        Please do not include characters like "\n" outside of text.

        [Summary of the situation]
        {basic_info['summary']}

        [Analyzed situations]
        {json.dumps(situation_data, ensure_ascii=False, indent=2)}

        [Fault Ratio]
        Plaintiff's fault ratio: {fault_rate:.2f}%

        Based on the above information, write a paragraph-long response from the counselor.
        Take a deep breath and step by step"""

        try:
            judgment_response = await self.client.chat.completions.create(
                model=Config.GPT_MODEL,
                messages=[{"role": "user", "content": judgment_prompt}],
                temperature=0.2
            )
            judgment = judgment_response.choices[0].message.content
        except Exception as e:
            print(f"Error in judgment generation: {e}")
            judgment = "판단 생성 중 오류가 발생했습니다."

        return {
            "status": "success",
            "method": "GPT",
            "data": {
                "id": request_id,
                "title": basic_info["title"],
                "stancePlaintiff": basic_info["stancePlaintiff"],
                "stanceDefendant": basic_info["stanceDefendant"],
                "summaryAi": basic_info["summary"],
                "judgement": judgment,
                "faultRate": fault_rate
            }
        }

async def process_request(request_data: dict) -> dict:
    analyzer = ConflictAnalyzer()
    result = await analyzer.analyze_content(
        content=request_data["content"],
        request_id=request_data.get("id", "id")
    )
    return result

# async def main():
#     request_data = {
#         "content": "최근에 새차를 뽑았습니다. 친구랑 근교 드라이브 가기로 하고 만났는데 친구가 제가 잠깐 차에서 내린 사이에 차안에 앉아서 담배를 피고 있더라구요?? 제가 미쳤냐고 왜 남의 차에서 담배 피냐고 화내니까’니네 아빠도 피우셨다며? 괜찮은줄 알았지’랍니다...상황 설명을 하자면 이 친구랑은 오래된 친구고 제가 비흡연자고 담배 엄청 싫어한다는거 잘 압니다. 전에 제가 몰던 차는 중고차였는데 제가 절대 차안에서 담배 못 피우게 했거든요. 이번에 처음으로 새차 뽑았고 옵션까지 다 해서 5천만원 쪼금 넘게 주고 구매했는데 새아버지가 절반 정도를 보태주셨어요. 능력 없는데 무리해서 산건 아니고 제가 전부 낼 능력 되서 고른차에요. 근데 새아버지가 딸 생기면 차한대 해주고 싶었다고 첨에는 전부 해주신다고 한걸 제가 부담스러워서 절반만 감사히 받았어요. 이런 사정까지 전부 아는 친구고요. 차 뽑고 첫 드라이브는 엄마랑 새아버지랑 다녀왔는데 새아버지가 흡연가고 하루 한갑정도 피실 정도로 담배를 사랑하시는 분이세요. 드라이브 중간에 비가 왔고 새아버지가 밖에 나가서 피운다고 하시길래 제가 비도 많이 오고 마음이 쓰이기도 했고 조만간 연차내고 내부 손세차 싹 할 생각이었어서 그냥 안에서 하셔도 된다 했어요. 솔직히 담배 냄새를 너무 싫어해서 좀 고민했지만 저랑 엄마에게 정말 잘해주시는 좋은 분이고 법적으로 부부도 되신지 한참이고 진짜 아빠로 받아들이기로 맘 먹어서 일부러 편해지려고 딸이니까 딸 차에서 편히 담배 태우시라고 한 것도 있었어요. 딱 한개피 피셨고 부모님 내려 드리고 친구를 바로 만나러 간건데 차안에 담배 냄새가 좀 났나봐요. 저보고 담배 피냐 하길래 방금 부모님 모시고 드라이브 다녀와서 그렇다고 아버지가 담배 태우시는거 알지 않냐고 했죠. 그리고 편의점에 들렀다나왔는데 차안에서 창문 열고 담배를 태우고 있더라구요. 제가 화를 냈더니 황당하단 얼굴로 “니네 아버지도 피우셨다길래 이제 너 차에서 담배 태워도 예민하지 않은줄 알았어. 미리 말하지 그랬어” 라며 저를 이상하게 모는거에요. 아니 이미 예전부터 차에서 담배 못 피게 했고 그리고 아버지랑 친구랑은 다르잖아요. 친아버지도 아니고 새아버지라 서로 예의 차리는 사이인것도 뻔히 알고 절반 보태주신것도 아는데 자기랑 새아버지랑 똑같이 담배를 펴도 된다고 생각 했다는게 놀라웠어요. 짜증은 났지만 손세차 할 예정이어서 앞으로는 그러지말라고 했는데 다른 친구들한테 제가 새차뽑고 유세만 엄청 떨다 갔다고, 담배 핀걸로 눈 까뒤집고 난리쳤다고 오바해서 얘기를 했대요. 다른 친구들은 제가 담배도 싫어하는 애고 거기다 새차였는데 담배핀 니가 문제라는 식으로 얘기하니까 더 오바하면서 새아버지는 줄담배 피우게 하고 자기한테만 못하게 한거라고 완전 거짓말까지 했더라고요. 저는 열받아서 손절하자 했고 그 친구도 자기가 더 어이없다며 자기가 먼저 손절 하려고 했다는 둥 여전히 뒤에서 제 욕을 하고 다녀요. 고작 담배 하나에 15년지기 친구를 손절했다면서요. 어떻게 생각하시나요? 제가 이상한건가요?",  # 실제 content
#         "nickname": "차주인",
#         "gender": "MALE",
#         "birth": "1996-05-04",
#         "id": "test_id"
#     }
    
#     result = await process_request(request_data)
#     print(json.dumps(result, ensure_ascii=False, indent=2))

# if __name__ == "__main__":
#     asyncio.run(main())