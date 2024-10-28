import json
from openai import ChatOpenAI
from some_module import ChatPromptTemplate, StrOutputParser, UserTextCategorise  # 필요한 모듈 임포트

async def summarize_and_evaluate_situation(ref_text: str) -> str:
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

    # Combined prompt template for summarization and evaluation
    combined_textbook = '''
    original text : {ref_text}
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
           - participants : Key participants in the event
           - result : The outcome or result of the event
           - time_frame : The time frame of the event, if mentioned
           - score : The score of the situation, ranging from 0 to 1 (0 being least important, 1 being most important)
    
    Output format:
    [
        {{"situation_summary": "", 
          "situation_case1": {{"event": "", "participants": "", "result": "", "time_frame": "", "score": 0.5}},
          "situation_case2": {{"event": "", "participants": "", "result": "", "time_frame": "", "score": 0.5}},
          ...
        }}
    ]
    '''

    # Generate the prompt with the provided ref_text
    prompt = ChatPromptTemplate.from_template(combined_textbook.format(ref_text=ref_text))

    # Chain connection
    chain = prompt | llm | StrOutputParser()

    # Generate response
    response = chain.invoke({
        'ref_text': ref_text
    })

    # Clean and parse JSON response
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    situations = json.loads(cleaned_data)

    return situations

async def behavior_evaluation(ref_text: str, situation_summary: str = None) -> str:
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

    # Behavior evaluation prompt template with optional situation_summary
    behavior_textbook = '''
    original text : {ref_text}
    {situation_summary_text}
    You are an evaluator who analyzes behaviors and attitudes expressed in the provided situation summary.
    Evaluate key behaviors, attitudes, impacts, and emotions based on the summary.
    
    Each behavior case should follow this format:
    - behavior_case1, behavior_case2, ... : Key behavior evaluation cases extracted from the summary.
       - attitude : The attitude of the speaker
       - behavior : The primary behavior or action performed by the speaker
       - impact : The significant impact of the behavior
       - emotion : The emotion expressed by the speaker during the interaction
       - score : The score of the behavior 0 to 1 (0 being least important, 1 being most important)
    
    Output format:
    [
        {{"behavior_case1": {{"attitude": "", "behavior": "", "impact": "", "emotion": "","score": 0.5}}},
          "behavior_case2": {{"attitude": "", "behavior": "", "impact": "", "emotion": "","score": 0.5}},
          ...
        }}
    ]
    '''

    # Insert situation_summary if provided
    situation_summary_text = f"summarized situation : {situation_summary}" if situation_summary else ""
    prompt = ChatPromptTemplate.from_template(behavior_textbook.format(ref_text=ref_text, situation_summary_text=situation_summary_text))

    # Chain connection
    chain = prompt | llm | StrOutputParser()

    # Generate response
    response = chain.invoke({
        'ref_text': ref_text
    })

    # Clean and parse JSON response
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    behaviors = json.loads(cleaned_data)

    return behaviors


