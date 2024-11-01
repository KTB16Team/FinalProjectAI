

async def question_categorize(ref_text) -> str:
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
        textbook = '''
        situation : {situation}
        당신은 situation의 내용을 요약하고 화자에 따라 분류하는 상황 분류기입니다.
        1. situation_summary
        : 사용자가 제공한 situation의 상황을 요약합니다.
            - 사용자는 A로 변경합니다.
            - 사용자와 대화하는 사람은 B로 변경합니다.
            - 그 외 등장 인물은 C,D,E와 같은 알파벳으로 변경합니다.
            - 요약할 때 각 사용자의 중요한 사건들이 사라지는 것을 지양해야합니다.
            - 최대한 객관적인 입장에서 내용을 요약합니다.
        2. individual_situation_summary_A,individual_situation_summary_B
        : 사용자가 제공한 situation 내용을 A와 B의 각각의 입장에서 요약합니다.
            - 각 사용자의 감정 변화를 반영하여 각 사용자의 입장에서 상황을 요약합니다.
            - 이때 반드시 사용자의 감정 변화에 영향을 미친 사건들이 사라지는 것을 막습니다.
            - 그리고 나누어진 내용을 각각 문자열로 individual_situation_summary_A,individual_situation_summary_B에 저장합니다.
            
        출력 방식은 다음과 같습니다:
        - json형식이다.
        - [{{"situation_summary" : "", "individual_situation_summary_" : ""}}, {{}}]
        
        '''
    # ChatPromptTemplate을 사용해 프롬프트 생성
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