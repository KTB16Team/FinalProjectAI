import json
import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any, List
import os

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
A: 당신 때문에 정말 화가나요! 약속 시간도 지키지 않고, 연락도 없고...
B: 죄송해요... 제가 일이 좀 바빠서...
A: 그 말도 벌써 세 번째예요. 이제는 믿기 힘들어요.
B: 아니, 이번엔 정말 급한 일이 있었어요! 다음부터는 꼭 시간 맞출게요.
A: 늘 그렇게만 말하고 바뀌는 건 없네요. 이제 지쳤어요.
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