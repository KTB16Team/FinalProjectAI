def situation_summary_GPT(text):
    return ""

def stt_model(link):
    return "text"

def generate_response(text):
    return "entities"

def test_response(text):
    '''
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
    - title": "제목",
      "stance_plaintiff": "원고(작성자) 입장",
      "stance_defendant": "피고(상대방) 입장",
      "summary_ai": "상황 요약문(AI)",
      "judgement": "판결문",
      "fault_rate": 57.8,}''
        "title": "사건 제목 (대화의 주제를 반영)}} ]
    '''
    prompt = ChatPromptTemplate.from_template(textbook)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'situation': ref_text
    })
    cleaned_data = response.replace('json', '')
    cleaned_data = cleaned_data.replace('```', '')
    cleaned_data = cleaned_data.strip()
    categorises = json.loads(cleaned_data)
    user_texts_list = [UserTextCategorise(**c) for c in categorises]

    return user_texts_list
'''    
1. 세트는 GPT를 사용하기 때문에 케이스별로 스코어가 매겨진다. 스코어를 종합적으로 계산해서 GPT를 사용하니깐. 
2. ML(감정 분석 어떻게 진행하실 예정?) 모델을 가져와서 어떻게/ 판단을 준 영향? 매핑(감정,태도(key)와 결과(value)(0-1) )
3. 3개의 프롬프트로 나눈다. 감정 점수 * 주요 원인(...)+ 행동 점수 * 주요 원인(추궁하는 것) + 상황 점수 * 주요원인 ( 0 )
3. 지각한 상황 * 감정 점수 (기다림,실망,믿음) + 지각한 상황 * 행동 원인 ( 추궁,실망,포기 )/ 지각한 상황 * 감정 점수 (미안,반성) + 지각한 상황 * 행동 원인 ( 사과,변명,반성 )
'''