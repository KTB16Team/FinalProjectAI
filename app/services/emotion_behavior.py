import json
import asyncio
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
load_dotenv()
# 스탠스 변화 부분 볼려면 인덱스 필요
class DialogueLine(BaseModel):
    index: int = Field(description="대화 문장의 인덱스")
    speaker: str = Field(description="화자")
    text: str = Field(description="대화 내용")

class StanceAction(BaseModel):
    index: int = Field(description="스탠스 변화가 발생한 대화 문장의 인덱스")
    dialogue_text: str = Field(description="해당 대화 문장")
    party: str = Field(description="대화 참여자 식별자")
    stance_classification: str = Field(description="태도 분류 (aggressive/defensive/avoidant/accepting 등)")
    score: float = Field(description="""행동의 과실 점수 (0-1):
    - 0에 가까울수록: 책임감 있고 건설적인 행동
    - 1에 가까울수록: 책임회피적이고 파괴적인 행동""")

class EmotionalImpact(BaseModel):
    from_party: str = Field(description="영향을 주는 참여자")
    to_party: str = Field(description="영향을 받는 참여자")
    impact_score: float = Field(description="감정 영향 점수 (-1 ~ +1)")
    emotional_state: List[str] = Field(description="주요 감정 상태")
    impact_description: str = Field(description="감정적 영향 설명")
    relevant_dialogue_indices: List[int] = Field(description="관련된 대화 인덱스")
#나중에 감정분석결과 A,B 입장 따로 스코어 매길거라
class EmotionalAnalysis(BaseModel):
    a_to_b_impact: EmotionalImpact = Field(description="A가 B에게 미치는 감정 영향")
    b_to_a_impact: EmotionalImpact = Field(description="B가 A에게 미치는 감정 영향")

class AnalysisResult(BaseModel):
    dialogue_lines: List[DialogueLine]
    stance_actions: List[StanceAction]
    emotional_impact: EmotionalAnalysis
    analysis_timestamp: str = Field(description="분석 수행 시간")

class RelationshipAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.2, #어느정도로 할건지?
            model="gpt-4o", #gpt-4는 되는데 4o는 안된다..?
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.parser = PydanticOutputParser(pydantic_object=AnalysisResult)

    def parse_dialogue(self, text: str) -> List[DialogueLine]:
        """원본 텍스트를 인덱스가 있는 대화 라인으로 파싱"""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        dialogue_lines = []
        
        for idx, line in enumerate(lines, 1):
            if ':' in line:
                speaker, content = line.split(':', 1)
                dialogue_lines.append(DialogueLine(
                    index=idx,
                    speaker=speaker.strip(),
                    text=content.strip()
                ))
        
        return dialogue_lines
#어떤 사람이 변화를 보여줬는지 party를 추출하는 것도 의미가 있을지?
    async def analyze_stance_changes(self, dialogue_lines: List[DialogueLine]) -> List[StanceAction]:
        prompt_template="""
        You are an expert in analyzing relationship dynamics and behavioral changes.

        Analyze the following conversation and identify points where a party's stance changes.
        
        Dialogue:
        {dialogue_lines}
                                                         
        For each stance change point:
        1. Identify clear changes in attitude or behavior
                                                         
        2. Classify the stance into one of these categories:
          - Aggressive (공격적): hostile, confrontational
          - Defensive (방어적): self-justifying, excuse-making
          - Avoidant (회피적): evasive, withdrawal
          - Accepting (수용적): understanding, acknowledging
          - Compromising (타협적): willing to meet halfway
          - Assertive (주장적): firm but not hostile

        3. Score each behavior (0-1) based on responsibility and fault:
          Lower scores (closer to 0):
          * Shows responsibility and accountability
          * Contributes to problem resolution
          * Demonstrates reasonable and appropriate behavior
          * Shows respect and understanding
          * Takes constructive actions
          
          Higher scores (closer to 1):
          * Avoids responsibility
          * Escalates conflicts
          * Shows inappropriate or harmful behavior
          * Demonstrates disrespect or lack of understanding
          * Takes destructive actions

        Return the analysis in the following JSON format:
        {{
            "stance_actions": [
                {{
                    "index": dialogue_line_index,
                    "dialogue_text": "exact text",
                    "party": "speaker",
                    "stance_classification": "stance type",
                    "score": "behavior_score(0-1)"
                }}
            ]
        }}
        Include only clear stance changes - not every dialogue line will represent a change point.
        Return strictly JSON output only. No explanation, no additional text.
        
        Take a deep breath and step by step.
        """       

        dialogue_text = "\n".join([f"{line.index}. {line.speaker}: {line.text}" for line in dialogue_lines])
        prompt = ChatPromptTemplate.from_template(template=prompt_template)

        try:
            response = await self.llm.agenerate([
                prompt.format_messages(dialogue_lines=dialogue_text)
            ])
            # 모델 변경시 안돌아갔던 부분 수정 코드
            response_text = response.generations[0][0].text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            print("Response text:", response.generations[0][0].text)
            print("Response text:", response)
            # print('errorcheck')
            result = json.loads(response_text)
            # print("1check")
            print(type(result))
            print(result)
            print([StanceAction(**action) for action in result['stance_actions']])
            result2 = [StanceAction(**action) for action in result['stance_actions']]
            print("last",type(result2))
            return result2
            # return [StanceAction(**action) for action in result['stance_actions']]

        except Exception as e:
            print(f"Stance analysis error: {str(e)}")
            raise
# 감정 점수를 저런식으로 나누는 기준을 애초부터 나눌지 아니면 알아서 판단하라 할지
    async def analyze_emotional_impact(self, dialogue_lines: List[DialogueLine], stance_actions: List[StanceAction]) -> EmotionalAnalysis:

        stance_info =[
            {
                "index": action.index,
                "party": action.party,
                "classification": action.stance_classification
            }
            for action in stance_actions
        ]
        print('fine')
        
        prompt_template="""
        You are an expert in analyzing emotional impacts in relationships

        Analyze the emotional impact between parties in the following conversation:

        Original Dialogue:
        {dialogue_text}

        Stance Classifications:
        {stance_info}

        Analyze the emotional impact in both directions (A to B and B to A):
                                                                                                                                                                    
        Consider both:
        1. The overall context from the original text
        2. The specific stance changes and their classifications       
                                                                                                        
        For each party (A and B), analyze:
        1. Emotional impact score (-1 to +1)
           - Severe negative: -1.0 to -0.7
           - Moderate negative: -0.7 to -0.3
           - Slight negative: -0.3 to 0
           - Slight positive: 0 to 0.3
           - Moderate positive: 0.3 to 0.7
           - Strong positive: 0.7 to 1.0

        2. For each direction (A→B and B→A), provide:
          - Impact score within the range
          - Key emotions experienced by the recipient
          - Detailed description of the emotional impact
          - Relevant dialogue indices showing this impact

        Return the analysis in the following JSON format:
        {{
          "emotional_analysis":{{
            a_to_b_impact:{{
              "from_party": "A",
              "to_party": "B",
              "impact_score": -1.0 to 1.0,
              "emotional_state": ["emotion1", "emotion2", ...],
              "impact_description": "detailed description",
              "relevant_dialogue_indices": [indices]
            }},
            a_to_b_impact:
            {{
              "from_party": "B",
              "to_party": "A",
              "impact_score": -1.0 to 1.0,
              "emotional_state": ["emotion1", "emotion2", ...],
              "impact_description": "detailed description",
              "relevant_dialogue_indices": [indices]
            }}
          }}
        }}

        Return strictly JSON output only. No explanation, no additional text.
        Take a deep breath and step by step.
        """
        dialogue_text = "\n".join([f"{line.index}. {line.speaker}: {line.text}" for line in dialogue_lines])
        prompt = ChatPromptTemplate.from_template(template=prompt_template)

        try:
            response = await self.llm.agenerate([
                prompt.format_messages(
                    dialogue_text=dialogue_text,
                    stance_info=json.dumps(stance_info, indent=2, ensure_ascii=False)
                )
            ])
            response_text = response.generations[0][0].text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            result = json.loads(response_text)
            analysis_data = result["emotional_analysis"]
            print(analysis_data)
            return EmotionalAnalysis(
                a_to_b_impact=EmotionalImpact(**analysis_data["a_to_b_impact"]),
                b_to_a_impact=EmotionalImpact(**analysis_data["b_to_a_impact"])
            )
            
        except Exception as e:
            print(f"Emotional analysis error: {str(e)}")
            print(f"Raw response: {response.generations[0][0].text if 'response' in locals() else 'No response'}")
            raise

    async def analyze(self, text: str) -> Dict:
      from datetime import datetime

      dialogue_lines = self.parse_dialogue(text)
      stance_results = await self.analyze_stance_changes(dialogue_lines)
      emotional_results = await self.analyze_emotional_impact(dialogue_lines, stance_results)
      
      # Pydantic 모델을 딕셔너리로 변환
      return {
          "dialogue_lines": [line.dict() for line in dialogue_lines],
          "stance_actions": [action.dict() for action in stance_results],
          "emotional_analysis": emotional_results.dict() if emotional_results else None,
          "analysis_timestamp": datetime.now().isoformat()
      }
  
async def test_analysis():
    start_time = time.time()
    test_data = """
      A: 당신 때문에 정말 화가나요! 약속 시간도 지키지 않고, 연락도 없고...
      B: 죄송해요... 제가 일이 좀 바빠서...
      A: 그 말도 벌써 세 번째예요. 이제는 믿기 힘들어요.
      B: 아니, 이번엔 정말 급한 일이 있었어요! 다음부터는 꼭 시간 맞출게요.
      A: 늘 그렇게만 말하고 바뀌는 건 없네요. 이제 지쳤어요.
    """

    analyzer = RelationshipAnalyzer()
    
    try:
        print("분석 시작...")
        result = await analyzer.analyze(test_data)

        print("\n대화 라인:")
        for line in result["dialogue_lines"]:
            print(f"{line['index']}. {line['speaker']}: {line['text']}")
        
        print("\n스탠스 변화 지점:")
        for action in result["stance_actions"]:
            print(f"\n대화 인덱스 {action['index']}:")
            print(f"대화 내용: {action['dialogue_text']}")
            print(f"변화 주체: {action['party']}")
            print(f"태도 분류: {action['stance_classification']}")
            print(f"과실 점수: {action['score']}")
        
        print("\n감정 영향 분석:")
        emotional = result["emotional_analysis"]
        
        print("\nA가 B에게 미친 영향:")
        a_to_b = emotional["a_to_b_impact"]
        print(f"영향 점수: {a_to_b['impact_score']}")
        print(f"감정 상태: {', '.join(a_to_b['emotional_state'])}")
        print(f"영향 설명: {a_to_b['impact_description']}")
        print(f"관련 대화 인덱스: {a_to_b['relevant_dialogue_indices']}")
        
        print("\nB가 A에게 미친 영향:")
        b_to_a = emotional["b_to_a_impact"]
        print(f"영향 점수: {b_to_a['impact_score']}")
        print(f"감정 상태: {', '.join(b_to_a['emotional_state'])}")
        print(f"영향 설명: b_to_a['impact_description']")
        print(f"관련 대화 인덱스: {b_to_a['relevant_dialogue_indices']}")

        end_time = time.time()

        # 소요 시간 계산
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_analysis())
