def situation_summary_GPT(text):
    analyzer=RelationshipAnalyzer()
    entities = analyzer.analyze(text)
    return entities
    # return ""

def stt_model(link):
    A = {'ai_stt':"stt_test_text"}
    return A

def generate_response(text):
    return "entities"


import json
#%%
import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import openai
import asyncio
from services.emotion_behavior_situation import RelationshipAnalyzer
#%%
# OPENAI_API_KEY
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(
    api_key = OPENAI_API_KEY,
)

analyzer = RelationshipAnalyzer
    
async def test_response(ref_text):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    # Combined prompt template for summarization and evaluation
    combined_textbook = '''
    situation : {situation}
    당신은 일반적인 사회 통념 관점에서 대화의 요약과 판결을 제공하는 중재자 판사입니다.
    다음 situation의 내용의 대화를 검토하고 모든 정보를 세부적으로 평가하여 아래에 요청된 정보들을 작성하세요.
    1. "title"
    : "사건 제목 (대화의 주제를 반영)"
    2. situation_summary
    : 사용자가 제공한 situation의 상황을 요약합니다.
        - 사용자는 A로 변경합니다.
        - 사용자와 대화하는 사람은 B로 변경합니다.
        - 그 외 등장 인물은 C,D,E와 같은 알파벳으로 변경합니다.
        - 요약할 때 각 사용자의 중요한 사건들이 사라지는 것을 지양해야합니다.
        - 최대한 객관적인 입장에서 내용을 요약합니다.
    2. stance_plaintiff
    : 사용자가 제공한 situation 내용을 A의 입장에서 요약합니다.
        - 각 사용자의 감정 변화를 반영하여 각 사용자의 입장에서 상황을 요약합니다.
        - 이때 반드시 사용자의 감정 변화에 영향을 미친 사건들이 사라지는 것을 막습니다
        - 원고(A)의 입장 설명 A의 시각에서 바라본 상황
        - A가 주장하는 사건의 핵심 요점
        - 대화 중 강조하는 감정과 이슈
    위의 내용을 stance_plaintiff에 저장합니다.
    3. stance_defendant
    : 사용자가 제공한 situation 내용을 B의 입장에서 요약합니다.
        - 각 사용자의 감정 변화를 반영하여 각 사용자의 입장에서 상황을 요약합니다.
        - 이때 반드시 사용자의 감정 변화에 영향을 미친 사건들이 사라지는 것을 막습니다
        - 피고(B)의 입장 설명 B의 시각에서 바라본 상황
        - B가 주장하는 사건의 핵심 요점
        - 대화 중 강조하는 감정과 이슈
    위의 내용을 stance_defendant에 저장합니다.
    4. judgement
    : 대화를 기반으로 한 최종 판결로, 각 주요 행동과 대화에서의 역할을 평가하여 판단한 결과. 사건의 결론과 필요한 경우 피고 및 원고에게 추천하는 행동 방안을 포함
    5. fault_rate
    : 과실 비율(숫자로 기재): 전체 사건에 대한 A의 과실 비율로, 분쟁에서 A의 책임 정도를 수치화한 것
    출력 방식은 다음과 같습니다:
    - json형식이다.
    - {{
      "stance_plaintiff": "원고(작성자) 입장",
      "stance_defendant": "피고(상대방) 입장",
      "summary_ai": "상황 요약문(AI)",
      "judgement": "판결문",
      "fault_rate": 57.8''
      "title": "사건 제목 (대화의 주제를 반영)}} ]
    '''
    prompt = ChatPromptTemplate.from_template(combined_textbook)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'situation': ref_text
    })
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    situations = json.loads(cleaned_data)

    return situations

async def test_response3(ref_text):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    # Combined prompt template for summarization and evaluation
    combined_textbook = '''
    situation : {situation}
    당신은 일반적인 사회 통념 관점에서 대화의 주제를 추출하는 역할입니다.
    다음 situation 내용의 대화를 검토하고 모든 정보를 세부적으로 평가하여 아래 요청된 정보들을 작성하세요.
    
    1. "situation_title"
       : "사건 제목 (대화의 주제를 반영)"
    
    2. situation_chunking
       - 주어진 situation의 내용을 검토하여 일반적인 사회 통념 관점에서 대화의 주요 주제를 추출하세요.
       - 대화 주제는 전체 내용에서 연속적으로 20% 이상을 차지하는 경우에만 주제로 분류하세요.
       - 여러 개의 주제가 있을 경우, 각 주제가 20% 이상일 때 모두 추출하세요.
       - 예시:
         - 외도(70%), 채무(30%) → 주제: 외도, 채무
         - 외도(70%), 폭력(20%), 채무(10%) → 주제: 외도, 폭력
       - 주제를 식별한 후, 해당 주제에 해당하는 원문 내용을 분리하여 origin_situation에 저장하세요.
       - 분리한 원문 텍스트는 절대 요약하거나 수정하지 마세요. 원문 그대로 사용해야 합니다.
       - 분리한 텍스트 파편들의 총합은 원문 전체와 동일한 길이를 가져야 합니다.
         즉, 원문에서 한 글자라도 누락해서는 안 됩니다.
    
    출력 형식은 json 배열입니다.
    예:
    [
      {
        "situation_chunking": "A주제명",
        "origin_situation": "A주제에 해당하는 원문 텍스트"
      },
      {
        "situation_chunking": "B주제명",
        "origin_situation": "B주제에 해당하는 원문 텍스트"
      }
    ]
    '''
    prompt = ChatPromptTemplate.from_template(combined_textbook)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'situation': ref_text
    })
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    situations = json.loads(cleaned_data)

    return situations


def test_response2(ref_text):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    # Combined prompt template for summarization and evaluation
    combined_textbook = '''
    situation : {situation}
    당신은 일반적인 사회 통념 관점에서 대화의 주제를 추출하는 역할입니다.
    다음 situation 내용의 대화를 검토하고 모든 정보를 세부적으로 평가하여 아래 요청된 정보들을 작성하세요.
    
    1. "situation_title"
       : "사건 제목 (대화의 주제를 반영)"
    
    2. situation_chunking
       - 주어진 situation의 내용을 검토하여 일반적인 사회 통념 관점에서 대화의 주요 주제를 추출하세요.
       - 대화 주제는 전체 내용에서 연속적으로 20% 이상을 차지하는 경우에만 주제로 분류하세요.
       - 여러 개의 주제가 있을 경우, 각 주제가 20% 이상일 때 모두 추출하세요.
       - 예시:
         - 외도(70%), 채무(30%) → 주제: 외도, 채무
         - 외도(70%), 폭력(20%), 채무(10%) → 주제: 외도, 폭력
       - 주제를 식별한 후, 해당 주제에 해당하는 원문 내용을 분리하여 origin_situation에 저장하세요.
       - 분리한 원문 텍스트는 절대 요약하거나 수정하지 마세요. 원문 그대로 사용해야 합니다.
       - 분리한 텍스트 파편들의 총합은 원문 전체와 동일한 길이를 가져야 합니다.
         즉, 원문에서 한 글자라도 누락해서는 안 됩니다.
    
    출력 형식은 json 배열입니다.
    예:
    [
      {
        "situation_chunking": "A주제명",
        "origin_situation": "A주제에 해당하는 원문 텍스트"
      },
      {
        "situation_chunking": "B주제명",
        "origin_situation": "B주제에 해당하는 원문 텍스트"
      }
    ]
    '''
    prompt = ChatPromptTemplate.from_template(combined_textbook)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'situation': ref_text
    })
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    situations = json.loads(cleaned_data)

    return situations

import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json

# ChatOpenAI와 Geminai 모델 초기화
chat_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, request_timeout=15)
geminai = ChatOpenAI(model="geminai-mini", temperature=0.3, request_timeout=15)

# 비동기 호출 함수
async def process_situation_with_fallback(ref_text):
    # 공통 프롬프트 정의
    prompt_template = '''
    situation : {situation}
    당신은 일반적인 사회 통념 관점에서 대화의 주제를 추출하는 역할입니다.
    다음 situation 내용의 대화를 검토하고 모든 정보를 세부적으로 평가하여 아래 요청된 정보들을 작성하세요.

    1. "situation_title"
       : "사건 제목 (대화의 주제를 반영)"

    2. situation_chunking
       - 주어진 situation의 내용을 검토하여 일반적인 사회 통념 관점에서 대화의 주요 주제를 추출하세요.
       - 대화 주제는 전체 내용에서 연속적으로 20% 이상을 차지하는 경우에만 주제로 분류하세요.
       - 여러 개의 주제가 있을 경우, 각 주제가 20% 이상일 때 모두 추출하세요.
       - 예시:
         - 외도(70%), 채무(30%) → 주제: 외도, 채무
         - 외도(70%), 폭력(20%), 채무(10%) → 주제: 외도, 폭력
       - 주제를 식별한 후, 해당 주제에 해당하는 원문 내용을 분리하여 origin_situation에 저장하세요.
       - 분리한 원문 텍스트는 절대 요약하거나 수정하지 마세요. 원문 그대로 사용해야 합니다.
       - 분리한 텍스트 파편들의 총합은 원문 전체와 동일한 길이를 가져야 합니다.
         즉, 원문에서 한 글자라도 누락해서는 안 됩니다.

    출력 형식은 json 배열입니다.
    예:
    [
      {
        "situation_chunking": "A주제명",
        "origin_situation": "A주제에 해당하는 원문 텍스트"
      },
      {
        "situation_chunking": "B주제명",
        "origin_situation": "B주제에 해당하는 원문 텍스트"
      }
    ]
    '''
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format_messages({"situation": ref_text})

    try:
        # Chat-GPT 비동기 호출
        response = await chat_gpt.acall(formatted_prompt)
        cleaned_data = json.loads(response.content.strip())
        return cleaned_data
    except Exception as e:
        print(f"Chat-GPT 오류 발생: {e}. Geminai로 대체 요청 시도.")
        try:
            # Geminai 비동기 호출
            response = await geminai.acall(formatted_prompt)
            cleaned_data = json.loads(response.content.strip())
            return cleaned_data
        except Exception as e:
            print(f"Geminai에서도 오류 발생: {e}")
            return None

# 여러 상황 비동기 처리
async def main():
    situations = [
        "Example situation text 1",
        "Example situation text 2",
        "Example situation text 3",
    ]

    # 모든 상황 병렬 처리
    results = await asyncio.gather(*[process_situation_with_fallback(sit) for sit in situations])

    # 결과 출력
    for idx, result in enumerate(results):
        if result:
            print(f"Situation {idx + 1} Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"Situation {idx + 1}: 처리 실패.")

if __name__ == "__main__":
    asyncio.run(main())

