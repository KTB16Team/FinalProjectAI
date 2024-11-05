import json
import asyncio
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import time

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
    impact_score: float = Field(description="감정 영향 점수")
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

class SituationCase(BaseModel):
    event: str = Field(description="이벤트 설명")
    participants: str = Field(description="참여자")
    result: str = Field(description="결과")
    time_frame: str = Field(description="시간 프레임")
    score: float = Field(description="중요도 점수 (0-1)")

class SituationSummary(BaseModel):
    situation_summary: str = Field(description="상황 요약")
    cases: List[SituationCase] = Field(description="상황 케이스들")

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

    async def summarize_and_evaluate_situation(self, dialogue_lines: List[DialogueLine]) -> SituationSummary:
        prompt_template = """
        original text : {dialogue_lines}
        
        You are an evaluator that performs both a summary of the situation and an objective analysis of each key event.
        
        1. situation_summary:
        - Summarize the situation provided by the user.
        - Replace the user with A, the person they are speaking to with B, and other individuals with C, D, E, etc.
        - Ensure that significant events involving each speaker are not omitted in the summary.
        - Provide an objective and neutral summary.
        
        2. situation evaluation:
        For each case extracted from the summarized situation, focus on objective events, excluding attitudes or emotions.
        
        Each situation case should follow this format:
        - situation_case1, situation_case2, ... : Key situation evaluation cases extracted from the summary.
            - event : A brief description of the event
            - participants : Key participants in the event.
            - result : The outcome or result of the event
            - time_frame : “Time frame” refers to the chronological order of events based on their context. For example, 
                            if the time frame for “situation_case1” is 1 and for “situation_case2” is 3, this indicates the sequential position of each event. 
                            The sequence is arranged according to the cause-and-effect relationship or the timeline in which each case occurs.
            - score : The score of the situation, ranging from 0 to 1 (0 being least important, 1 being most important)
        
        Return only the following JSON format without any additional text:
        {{
        "situation_summary": "complete situation summary",
        "situation_cases": [
            {{
                "event": "event description",
                "participants": "A, B",
                "result": "event result",
                "time_frame": "1",
                "score": 0.8
            }},
            ...
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
            
            response_text = response.generations[0][0].text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text)
            # print(result) 
            cases = [SituationCase(**case) for case in result.get("situation_cases", [])]
            summary = SituationSummary(
                situation_summary=result["situation_summary"],
                cases=cases
            )
            # print(summary)
            return summary
        
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러: {str(e)}")
            print(f"원본 응답: {response_text}")
            raise
        except Exception as e:
            print(f"상황 분석 에러: {str(e)}")
            raise
        
    async def analyze_stance_changes(self, dialogue_lines: List[DialogueLine], situation_results: SituationSummary) -> List[StanceAction]:
        
        prompt_template="""
        You are an expert in analyzing relationship dynamics and behavioral changes.

        Analyze the following conversation and identify points where a party's stance changes.
        
        Situation Summary:
        {situation_summary}

        Key Events:
        {situation_cases}

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
        situation_summary_text = situation_results.situation_summary
        situation_cases_text = "\n".join([
            f"- Event: {case.event}, Participants: {case.participants}, Result: {case.result}, Time Frame: {case.time_frame}, Score: {case.score}"
            for case in situation_results.cases
        ])

        prompt = ChatPromptTemplate.from_template(template=prompt_template)

        try:
            response = await self.llm.agenerate([
                prompt.format_messages(
                    situation_summary = situation_summary_text,
                    situation_cases = situation_cases_text,
                    dialogue_lines=dialogue_text)
            ])
            # 모델 변경시 안돌아갔던 부분 수정 코드
            response_text = response.generations[0][0].text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            # print("Response text:", response.generations[0][0].text)
            # print("Response text:", response)
            # print('errorcheck')
            result = json.loads(response_text)
            # print("1check")
            # print(type(result))
            # print(result)
            # print([StanceAction(**action) for action in result['stance_actions']])
            result2 = [StanceAction(**action) for action in result['stance_actions']]
            # print("last",type(result2))
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
        
        prompt_template="""
        You are an expert in analyzing emotional impacts in relationships.

        Analyze the emotional impact between parties in the following conversation:

        Original Dialogue:
        {dialogue_text}

        Stance Classifications:
        {stance_info}

        Analyze the emotional impact in both directions (A to B and B to A):
                                                                                                                                                                    
        Consider both:
        1. The overall context from the original text
        2. The specific stance changes and their classifications       
                                                                                                        
        Scoring Mechanism:
          - Positive and constructive actions should be reflected with a high score close to 1, while negative or destructive actions result in a score closer to 0
          - Maintain scores between 0.1 and 1 only. 
          - Ensure no behavior score is exactly 0, preventing division errors errors.

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
              "impact_score": 0.1 to 1,
              "emotional_state": ["emotion1", "emotion2", ...],
              "impact_description": "detailed description",
              "relevant_dialogue_indices": [indices]
            }},
            a_to_b_impact:
            {{
              "from_party": "B",
              "to_party": "A",
              "impact_score": 0.1 to 1,
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
            # print(analysis_data)
            return EmotionalAnalysis(
                a_to_b_impact=EmotionalImpact(**analysis_data["a_to_b_impact"]),
                b_to_a_impact=EmotionalImpact(**analysis_data["b_to_a_impact"])
            )
            
        except Exception as e:
            print(f"Emotional analysis error: {str(e)}")
            print(f"Raw response: {response.generations[0][0].text if 'response' in locals() else 'No response'}")
            raise
    
    async def calculate_fault_ratio(self, situation_results: SituationSummary, stance_results: List[StanceAction], emotional_analysis: EmotionalAnalysis) -> Dict[str, float]:
        """
        Calculate fault ratio ensuring the total equals 100%.
        """

        situation_score = sum(case.score for case in situation_results.cases) / len(situation_results.cases) if situation_results.cases else 1.0
        behavior_score = sum(action.score for action in stance_results) / len(stance_results) if stance_results else 1.0
        
        emotion_score_a_to_b = max(0.01, (1 - emotional_analysis.a_to_b_impact.impact_score))
        emotion_score_b_to_a = max(0.01, (1 - emotional_analysis.b_to_a_impact.impact_score))
        
        fault_score_a = situation_score * behavior_score * emotion_score_a_to_b
        fault_score_b = situation_score * behavior_score * emotion_score_b_to_a

        total_score = fault_score_a + fault_score_b
        fault_ratio_a = fault_score_a / total_score
        fault_ratio_b = fault_score_b / total_score
        
        return {
            "A": round(fault_ratio_a, 2),
            "B": round(fault_ratio_b, 2)
        }
    
    async def generate_judgment_statement(self, situation_results: SituationSummary, fault_ratios: Dict[str, float], stance_results: List[StanceAction]) -> str:
        prompt_template = """
        You are an impartial arbitrator delivering a final judgment in a dispute. Given the following details:

        Situation Summary:
        {situation_summary}

        Key Situation Cases:
        {situation_cases}

        Fault Ratios:
        - Participant A's Fault Ratio: {a_fault_ratio}%
        - Participant B's Fault Ratio: {b_fault_ratio}%

        Behavioral Changes:
        {stance_changes}

        Based on the above data, deliver a final judgment statement that:
        1. Summarizes the perspectives of both A and B based on the provided summaries.
        2. Clearly outlines the culpability percentages and explains the reasoning behind them.
        3. Concludes with an objective final statement on the overall fault distribution and resolution advice if applicable.
        4. Please print in Korean.

        Final Output Format:
        "Judgment Statement: complete judgment text"
        """

        situation_summary_text = situation_results.situation_summary
        situation_cases_text = "\n".join([
            f"- Event: {case.event}, Participants: {case.participants}, Result: {case.result}, Time Frame: {case.time_frame}, Score: {case.score}"
            for case in situation_results.cases
        ])

        stance_changes_text = "\n".join([
            f"- Line {action.index}: {action.dialogue_text} (Stance: {action.stance_classification}, Score: {action.score})"
            for action in stance_results
        ])

        prompt = ChatPromptTemplate.from_template(template=prompt_template)

        try:
            response = await self.llm.agenerate([
                prompt.format_messages(
                    situation_summary=situation_summary_text,
                    situation_cases=situation_cases_text,
                    a_fault_ratio=fault_ratios["A"],
                    b_fault_ratio=fault_ratios["B"],
                    stance_changes=stance_changes_text
                )
            ])
            response_text = response.generations[0][0].text.strip()

            return response_text.replace("Judgment Statement:", "").strip()
        except Exception as e:
            print(f"Judgment statement generation error: {str(e)}")
            raise 


    async def analyze(self, text: str) -> Dict:
      from datetime import datetime

      dialogue_lines = self.parse_dialogue(text)
      situation_results = await self.summarize_and_evaluate_situation(dialogue_lines)#추가
      stance_results = await self.analyze_stance_changes(dialogue_lines, situation_results)
      emotional_results = await self.analyze_emotional_impact(dialogue_lines, stance_results)
      fault_ratios = await self.calculate_fault_ratio(situation_results, stance_results, emotional_results)
      judgement = await self.generate_judgment_statement(situation_results, fault_ratios, stance_results)
      # Pydantic 모델을 딕셔너리로 변환
      return {
          "dialogue_lines": [line.dict() for line in dialogue_lines],
          "situation_summary": situation_results.dict(),
          "stance_actions": [action.dict() for action in stance_results],
          "emotional_analysis": emotional_results.dict() if emotional_results else None,
          "fault_ratios": fault_ratios,
          "judgement": judgement,
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
        
        situation_summary = result["situation_summary"]
        print("\n상황 요약:")
        print(f"{situation_summary['situation_summary']}")
        
        print("\n상황 케이스들:")
        for case in situation_summary["cases"]:
            print(f"- 이벤트: {case['event']}")
            print(f"  참여자: {case['participants']}")
            print(f"  결과: {case['result']}")
            print(f"  시간 프레임: {case['time_frame']}")
            print(f"  상황 점수: {case['score']}\n")

        print("\n스탠스 변화 지점:")
        for action in result["stance_actions"]:
            print(f"\n액션 인덱스 {action['index']}:")
            print(f"액션 내용: {action['dialogue_text']}")
            print(f"변화 주체: {action['party']}")
            print(f"태도 분류: {action['stance_classification']}")
            print(f"행동 평가 점수: {action['score']}")
        
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

        print("\n과실 비율:")
        print(f"A의 과실 비율: {result['fault_ratios']['A'] * 100:.2f}%")
        print(f"B의 과실 비율: {result['fault_ratios']['B'] * 100:.2f}%")

        print("\n판결문:")
        print(result["judgement"])

        end_time = time.time()

        # 소요 시간 계산
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_analysis())
