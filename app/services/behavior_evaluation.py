import json
import asyncio
import openai
import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(
    api_key = OPENAI_API_KEY,
)
async def behavior_evaluation(ref_text: str, situation_summary: str = None) -> Dict[str, List[Dict]]:
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    system_prompt = """You are an expert in analyzing communication behaviors and attitudes. 
    Focus on specific actions, speech patterns, and attitudes in the conversation. 
    Provide analysis in the exact JSON format specified."""

    user_prompt = f"""Analyze the following conversation, focusing on specific behaviors and attitudes:

TEXT TO ANALYZE:
{ref_text}

{situation_summary if situation_summary else ''}

Please analyze each participant's behaviors using these criteria:

1. Speech Patterns:
   - Tone (aggressive/defensive/neutral/cooperative)
   - Expression Style (direct/indirect/emotional/rational)
   - Intensity (1-5 scale)
   - Persistence (one-time/repetitive)

2. Conversational Attitude:
   - Listening Level
   - Response Style
   - Responsibility Acceptance
   - Resolution Willingness

3. Behavior Score (0-1):
   Score each behavior based on how it affects responsibility and fault in the situation:
   - Higher scores (closer to 1): 
     * Shows responsibility and accountability
     * Contributes to problem resolution
     * Demonstrates reasonable and appropriate behavior
     * Shows respect and understanding
     * Takes constructive actions
   
   - Lower scores (closer to 0):
     * Avoids responsibility
     * Escalates conflicts
     * Shows inappropriate or harmful behavior
     * Demonstrates disrespect or lack of understanding
     * Takes destructive actions

Provide your analysis in this exact JSON format:
{{
  "behavior_cases": [
    {{
      "case_id": "1",
      "speaker": "speaker identifier",
      "action": "specific action/speech",
      "speech_characteristics": {{
        "tone": "tone description",
        "expression_style": "style description",
        "intensity": 3,
        "persistence": "one-time/repetitive"
      }},
      "attitude": {{
        "listening": "listening level",
        "response_type": "response style",
        "responsibility": "responsibility level",
        "resolution_willingness": "willingness level"
      }},
      "score": 0.5
    }}
  ]
}}
Take a deep breath, and work on this step by step."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        result = response.choices[0].message.content.strip()
        
        try:
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러: {str(e)}")
            print(f"받은 응답: {result}")
            raise

    except Exception as e:
        print(f"API 요청 에러: {str(e)}")
        raise

async def test_behavior_evaluation():
    test_data = """
A: 당신 때문에 정말 ... 약속 시간도 지키지 않고, 연락도 없고...{공격} => 0.8 
B: 죄송해요... 제가 일이 좀 바빠서...{수용}
A: 그 말도 벌써 세 번째예요. 이제는 믿기 힘들어요.{공격}
B: 아니, 이번엔 정말 급한 일이 있었어요! 다음부터는 꼭 시간 맞출게요.{지각}
A: 늘 그렇게만 말하고 바뀌는 건 없네요. 이제 지쳤어요.{지각}
B: 근데 당신도 늦잖아요.{지각,내로남불}
A: 그래서요? 당신이랑 나랑 같아요? 집에 갈래요{지각,내로남불}
B: 너무한거 아닌가요? 내로남불 너무 심한데? 당신 이런 사람이였어?그리고 어제 당신 야근 한다면서 왜 바다에 갔어?{지각,내로남불,외도}
A: 이제는 미행도 하나요? 그냥 업무 출장차 나갔다온거에요{내로남불,외도}
B: 바다로? 말같지 않은 소리 하지 말고 당신 업무랑 바다가 무슨 상관이지?
A: 믿지도 못하면서 왜 물어보신거죠? 우리 머리식히고 다시만나요. 저 갈게요
    """

    try:
        print("대화 행위 분석 시작...")
        result = await behavior_evaluation(test_data)
        print("\n분석 결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_behavior_evaluation())

'''    
1. 세트는 GPT를 사용하기 때문에 케이스별로 스코어가 매겨진다. 스코어를 종합적으로 계산해서 GPT를 사용하니깐. 
2. ML(감정 분석 어떻게 진행하실 예정?) 모델을 가져와서 어떻게/ 판단을 준 영향? 매핑(감정,태도(key)와 결과(value)(0-1) )
3. 3개의 프롬프트로 나눈다. 감정 점수 * 주요 원인(...)+ 행동 점수 * 주요 원인(추궁하는 것) + 상황 점수 * 주요원인 ( 0 )
3. 지각한 상황 * 감정 점수 (기다림,실망,믿음) + 지각한 상황 * 행동 원인 ( 추궁,실망,포기 )/ 지각한 상황 * 감정 점수 (미안,반성) + 지각한 상황 * 행동 원인 ( 사과,변명,반성 )
'''