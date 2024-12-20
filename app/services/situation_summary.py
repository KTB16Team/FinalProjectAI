import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from services.emotion_behavior_situation import RelationshipAnalyzer
from core.logging import setup_logger

# 로거 설정
logger = setup_logger()

# Load OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Analyzer 객체
analyzer = RelationshipAnalyzer()

def situation_summary_GPT(text):
    entities = analyzer.analyze(text)
    return entities

def stt_model(link):
    return {'ai_stt': "stt_test_text"}

def generate_response(text):
    return "entities"

def test_response(ref_text):
    if isinstance(ref_text, str):
        logger.info("ref_text is a string, converting to dict with content.")
        ref_text = {"content": ref_text}

    content = ref_text.get('content', '')
    prompt = f"""
    situation : {content}
    당신은 일반적인 사회 통념 관점에서 대화의 요약과 판결을 제공하는 중재자 판사입니다.
    다음 situation의 내용의 대화를 검토하고 모든 정보를 세부적으로 평가하여 아래에 요청된 정보들을 작성하세요.
    1. "title"
    : "사건 제목 (대화의 주제를 반영)"
    2. summaryAi
    : 사용자가 제공한 situation의 상황을 요약합니다.
        - 사용자는 A로 변경합니다.
        - 사용자와 대화하는 사람은 B로 변경합니다.
        - 그 외 등장 인물은 C,D,E와 같은 알파벳으로 변경합니다.
        - 요약할 때 각 사용자의 중요한 사건들이 사라지는 것을 지양해야합니다.
        - 최대한 객관적인 입장에서 내용을 요약합니다.
    3. stancePlaintiff
    : 사용자가 제공한 situation 내용을 A의 입장에서 요약합니다.
        - 각 사용자의 감정 변화를 반영하여 각 사용자의 입장에서 상황을 요약합니다.
        - 이때 반드시 사용자의 감정 변화에 영향을 미친 사건들이 사라지는 것을 막습니다
        - 원고(A)의 입장 설명 A의 시각에서 바라본 상황
        - A가 주장하는 사건의 핵심 요점
        - 대화 중 강조하는 감정과 이슈
    4. stanceDefendant
    : 사용자가 제공한 situation 내용을 B의 입장에서 요약합니다.
        - 각 사용자의 감정 변화를 반영하여 각 사용자의 입장에서 상황을 요약합니다.
        - 이때 반드시 사용자의 감정 변화에 영향을 미친 사건들이 사라지는 것을 막습니다
        - 피고(B)의 입장 설명 B의 시각에서 바라본 상황
        - B가 주장하는 사건의 핵심 요점
        - 대화 중 강조하는 감정과 이슈
    5. judgement
    : 대화를 기반으로 한 최종 판결로, 각 주요 행동과 대화에서의 역할을 평가하여 판단한 결과. 사건의 결론과 필요한 경우 피고 및 원고에게 추천하는 행동 방안을 포함
    6. faultRate
    : 과실 비율(숫자로 기재): 전체 사건에 대한 A의 과실 비율로, 분쟁에서 A의 책임 정도를 수치화한 것
    출력 방식은 다음과 같습니다:
    - json형식이다.
    - {{
      "title": "사건 제목",
      "stancePlaintiff": "원고(작성자) 입장",
      "stanceDefendant": "피고(상대방) 입장",
      "summaryAi": "상황 요약문(AI)",
      "judgement": "판결문",
      "faultRate": 57.8}}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content
        logger.info(f"GPT Response(raw): {content}")  # GPT의 원문 응답

        # JSON 파싱 시도
        try:
            result = json.loads(content)

            # GPT 응답 내용을 자세히 로그 출력
            logger.info(f"GPT Response(parsed): {json.dumps(result, ensure_ascii=False, indent=2)}")

            # 필요한 필드들 추출 (존재하지 않으면 기본값으로 처리)
            title = result.get("title", "")
            stancePlaintiff = result.get("stancePlaintiff", "")
            stanceDefendant = result.get("stanceDefendant", "")
            summaryAi = result.get("summaryAi", "")
            judgement = result.get("judgement", "")

            # faultRate는 float형으로 변환 시도
            faultRate_value = result.get("faultRate", 0)
            if not isinstance(faultRate_value, (int, float)):
                # 문자열 형태일 경우 float 변환
                try:
                    faultRate_value = float(faultRate_value)
                except ValueError:
                    # 변환 실패 시 기본값 0.0 사용
                    faultRate_value = 0.0

            return {
                "status": True,
                "accessKey": "dwqdq",
                "id": ref_text.get("id", 0),
                "title": title,
                "stancePlaintiff": stancePlaintiff,
                "stanceDefendant": stanceDefendant,
                "summaryAi": summaryAi,
                "judgement": judgement,
                "faultRate": faultRate_value
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error. Raw response: {content}")
            return {
                "status": False,
                "accessKey": "dwqdq",
                "id": ref_text.get("id", 0),
                "title": "",
                "stancePlaintiff": "",
                "stanceDefendant": "",
                "summaryAi": "",
                "judgement": "JSON 파싱 오류",
                "faultRate": 0.0
            }

    except Exception as e:
        logger.error(f"Error in GPT request: {e}")
        return {
            "status": False,
            "accessKey": "dwqdq",
            "id": ref_text.get("id", 0),
            "title": "",
            "stancePlaintiff": "",
            "stanceDefendant": "",
            "summaryAi": "",
            "judgement": f"오류: {str(e)}",
            "faultRate": 0.0
        }
