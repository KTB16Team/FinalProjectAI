#%%
import sys
import os
import pickle
import json
import ast
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import openai
import asyncio

#%%
# OPENAI_API_KEY
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(
    api_key = OPENAI_API_KEY,
)
#%%
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
           - participants : Key participants in the event.
           - result : The outcome or result of the event
           - time_frame : “Time frame” refers to the chronological order of events based on their context. For example, 
                         if the time frame for “situation_case1” is 1 and for “situation_case2” is 3, this indicates the sequential position of each event. 
                         The sequence is arranged according to the cause-and-effect relationship or the timeline in which each case occurs.
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
    prompt = ChatPromptTemplate.from_template(combined_textbook)

    # Chain connection
    chain = prompt | llm | StrOutputParser()

    # Generate response
    response = chain.invoke({
        'ref_text': ref_text
    })

    # Clean and parse JSON response
    print(response)
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
    prompt = ChatPromptTemplate.from_template(behavior_textbook)

    # Chain connection
    chain = prompt | llm | StrOutputParser()

    # Generate response
    response = chain.invoke({
        'ref_text': ref_text,
        'situation_summary_text': situation_summary_text
    })

    # Clean and parse JSON response
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    behaviors = json.loads(cleaned_data)

    return behaviors

async def emotion_evaluation(ref_text: str, situation_summary: str = None) -> str:
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

    # Emotion evaluation prompt template
    emotion_textbook = '''
    Analyze the emotions in each dialogue entry below. For each entry, identify the speaker, calculate the emotion scores, and determine whether each emotion was given to the other person or received from them. Each detected emotion should be scored on a scale from 0 to 1, where 0 means no emotion and 1 means a very strong emotion.

    Dialogue:
    {dialogue_text}
    {situation_summary_text}
    Instructions:
    1. For each dialogue entry, detect the emotions conveyed in the text.
    2. Identify each emotion as either given to the other person or received from them.
    3. Assign a score between 0 and 1 for each detected emotion in both categories.
    4. Record the speaker (A or B) and the dialogue entry number.

    Output format (JSON):
    [
        {{
            "entry": 1,
            "speaker": "A",
            "given_emotions": {{
                "joy": 0.8,
                "trust": 0.0,
                "fear": 0.6,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.5
            }},
            "received_emotions": {{
                "joy": 0.8,
                "trust": 0.2,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0
            }}
        }},
        {{
            "entry": 2,
            "speaker": "B",
            "given_emotions": {{
                "joy": 0.6,
                "trust": 0.8,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.3
            }},
            "received_emotions": {{
                "joy": 0.0,
                "trust": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0
            }}
        }},
        {{
            "entry": 3,
            "speaker": "A",
            "given_emotions": {{
                "joy": 0.5,
                "trust": 0.9,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.4
            }},
            "received_emotions": {{
                "joy": 0.5,
                "trust": 0.9,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0
            }}
        }}
    ]
    '''

    # Prompt creation
    situation_summary_text = f"summarized situation : {situation_summary}" if situation_summary else ""
    prompt = ChatPromptTemplate.from_template(emotion_textbook)

    # Chain connection
    chain = prompt | llm | StrOutputParser()

    # Generate response
    response = chain.invoke({
        'dialogue_text': ref_text,
        'situation_summary_text': situation_summary_text
    })

    # Clean and parse JSON response
    cleaned_data = response.replace('json', '').replace('```', '').strip()
    emotions = json.loads(cleaned_data)

    return emotions
#%%

test_Case1 = '''
A: 또 30분이나 늦었네. 우리가 몇 시에 만나기로 했는지 기억해?

B: 차가 막혀서 어쩔 수 없었어. 미리 연락했잖아. 정말 빨리 오려고 했다고.

A: 그거야 그렇지. 근데 벌써 몇 번째야? 매번 이렇게 기다리게 하면, 나도 지치는 거 몰라?

B: 미안하다고 했잖아. 일부러 늦은 것도 아니고… 상황이 어쩔 수 없었다니까.

A: ‘어쩔 수 없었다’는 말 이제 그만 좀 듣고 싶어. 너는 항상 변명만 늘어놓고, 내가 매번 이해해야 한다고만 생각하는 거 같아.

B: 아니야, 진짜 사정이 있었으니까 그런 거잖아. 이해해줄 수 없는 거야?

A: 그럼, 네가 계속 이렇게 반복해서 늦을 때마다 내가 이해만 해야 하는 거야? 네가 얼마나 기다리게 하는 건지, 그동안 내가 어떻게 느끼는지 알기나 해?

B: 나도 불편한데… 진짜 상황이 그랬으니까 어쩔 수 없다고 이해해줬으면 좋겠어. 기다리는 게 싫으면 딱히 억지로 기다릴 필요도 없었잖아.

A: 억지로 기다렸다고 생각해? 그게 무슨 말이야? 너를 만나고 싶어서, 시간 맞춰 나오고 기다린 건데… 그게 억지라니. 그러면 차라리 네가 시간을 더 맞추려고 노력했어야지.

B: 무슨 소리야, 나도 노력했어. 매번 늦고 싶은 게 아니라고. 근데 내가 뭘 할 수 있었겠어? 차가 막혀서 그랬는데 내가 어떻게 했어야 했는데?

A: 시간 약속이 중요하다면, 애초에 일찍 출발하거나, 방법을 미리 생각해봤겠지. 언제나 ‘어쩔 수 없었다’는 말만 하니까 신뢰가 떨어지는 거잖아.

B: 그래, 내가 매번 늦었던 건 아니잖아. 오늘은 상황이 안 좋았던 거지. 그리고 그런 말을 듣는 것도 나로선 부담스러워. 네가 날 전혀 이해 안 해주는 것 같아서…

A: 네가 그러는 것처럼 나도 사정이 있겠지. 근데 내가 항상 네 입장을 이해해야 해? 나도 네가 기다리는 거 신경 써줬으면 좋겠는데 그게 그렇게 힘든 거야?

B: 너도 내가 왜 늦었는지 이해해줘야지. 내 입장도 생각해주고, 상황이 어쩔 수 없는 건 받아들일 수 있는 거잖아. 계속 이렇게 불만을 쌓아가는 게 너한테도 좋은 건 아니라고 생각해.

A: 난 네가 늦는 거보다 항상 이유를 찾아내서 자기 입장만 고집하는 게 더 답답해. 그럴 때마다 내가 무슨 생각을 하는지 네가 전혀 관심 없어 보여.

B: 그러면 매번 무조건 미안하다고만 할 수는 없잖아. 나도 진짜 노력을 하고 있는데, 네가 자꾸 이 문제로 날 몰아붙이면 솔직히 화도 나고 힘들어.

A: 이해 못할 것도 없겠지, 근데 네가 나한테 더 관심을 두고 노력했다면 이런 일이 없었을 거야. 우리는 약속할 때마다 반복되는 일이니까 그만큼 중요하게 생각해줬으면 좋겠어.

B: 자꾸 이렇게 나한테만 불만을 쌓아놓으면 그게 정말 좋은 관계를 유지할 수 있는 방법이라고 생각해?

A: 난 노력하라고 계속 말하는 거야. 그렇지 않고서는 넌 더 이해 못할 거 같아서… 이런 상황이 계속 반복되면 결국 우리 둘 다 지칠 거야.

B: 정말 내가 널 이해 못 한다는 말밖에 안 들리네. 난 이게 내 잘못인 줄 알고 미안해하려고 해도 네가 그럴수록 더 부담돼.

A: 이해 못하는 게 아니잖아, 그냥 내가 말한 것처럼 실수하고 있는 걸 알고 그 부분을 조심하자는 건데… 그런 게 어렵다고 생각해?

B: 나도 최대한 네 입장을 생각하겠지만, 그럼 너도 내 상황을 한 번만이라도 이해해주는 게 그렇게 어려워?

A: 그럼 네 말대로 무조건 이해해줄게. 네가 시간을 아무리 늦어도 다 이해하고 받아들이면 되겠네. 그래, 이렇게 하면 해결이 되긴 할 거야.

B: 진짜로 그렇게 생각하는 거야? 그냥 네가 원하는 대로 난 이해만 하고 넘어가야 한다는 거야?

A: 뭐, 그게 서로를 위해서라면 할 수도 있지. 문제는 네가 먼저 그렇게 하려고 하지 않았다는 거잖아. 그래도 난 널 배려하려고 노력했어.

B: 배려는 양쪽이 다 해야 하는 거야. 내 입장에서도 너의 기분을 생각하려고 했는데, 네가 내 마음을 이해해주지 않으니까 힘들다고 느껴지는 거잖아.

A: 그렇다면 난 매번 네가 하는 걸 받아들여야 하고, 너는 그대로 남아도 된다는 거네? 그게 무슨 관계야?

B: 너도 참… 그렇게 모든 게 네 말대로만 가야 하는 거야? 난 노력하려는 것 자체를 말하는 건데.

A: 그러면 내 말대로 안 할 거야? 어떻게 해야 네가 이해할 건데?

B: 그건 서로 이해하고 맞춰가는 거야. 계속 서로가 받아들일 수 있게… 말로만 이해하라는 건 다 의미 없어져.
A: 그럼 네 말대로 난 그냥 너한테 맞추고, 계속 이해만 해야 하는 거네? 그래, 그렇게 하면 되겠네. 네가 늦고, 매번 변명하고, 난 그냥 받아들이고… 이게 다야?

B: 왜 그렇게 비꼬는 거야? 나도 이 상황이 힘들다고. 내 입장은 생각 안 하고 무조건 내 잘못만 강조하는 거 정말 지쳤어.

A: 지쳤다고? 나는 안 지쳤겠어? 매번 같은 이유로 기다리고, 화내면 네 입장 생각해줘야 하고, 결국 내가 이해하라는 거잖아. 이게 반복되니까 더 힘들어.

B: 그래서 너는 모든 잘못이 다 내 탓이라고 생각하는 거야? 그럼 이제 뭐 어쩌자는 거지? 계속 이 얘기만 하고 있을 거야?

A: 내가 내 입장을 말하면 왜 매번 네가 억울해 하는지 모르겠어. 그렇게 힘들면 처음부터 시간을 맞췄어야지. 애초에 약속 시간 안 지키는 게 문제인 거잖아.

B: 그럼 넌 한 번이라도 내가 왜 늦는지 이해하려고 했어? 차가 막혔고, 진짜 어쩔 수 없는 상황도 있었는데… 날 한 번이라도 이해하려고 노력은 했어?

A: 이해해주려고 했지. 근데 너는 왜 매번 똑같은 핑계야? 차가 막혔다는 변명으로 계속 반복되는 게 너는 이해가 돼?

B: 알겠어, 알겠다고. 그러면 내가 항상 틀렸고, 네가 옳다는 말이야? 난 진짜 지쳤어, 더 이상 말하기도 싫어.

A: 그래, 그럼 더 얘기 안 할게. 어차피 얘기해봤자 네가 듣지 않는 것 같으니까.

B: 어, 나도 그래. 네 말 다 들어봤으니까… 이제 뭐 더 말할 것도 없겠네.

A: 네가 이해하든 말든 상관없어. 이런 식으로 하면 우리 그냥 끝나는 거야. 이 얘기를 계속해도 달라질 게 없으니까.

B: 좋아, 그럼 여기서 끝내. 나도 이제는 진짜 이게 맞는 건가 싶어. 너도 집에 가.

A: 좋아, 나도 집에 갈게. 이젠 할 말도 없어.

B: 그래, 가라.

A: 어차피 너도 나도 해결할 생각 없으니까, 그냥 끝내자.

B: 그렇게 말하고 싶으면 그렇게 하자고.

A: 그럼, 나 간다.

B: 어, 나도 갈 거니까 더 이상 말하지 마.'''


#%%
# 파일 열고 읽기
# with open("test_Case1.txt", "r") as file:
#     content = file.read()  # 파일 전체 읽기
#     print(content)

content = test_Case1
#%%
A = asyncio.run(summarize_and_evaluate_situation(content))
B = asyncio.run(behavior_evaluation(content,A[0]['situation_summary']))
C = asyncio.run(emotion_evaluation(content,A[0]['situation_summary']))
#%%

# print(A)
keys_list = list(A[0].keys())
print(A[0]['situation_summary'])
print(keys_list)
print(A[0][keys_list[1]])
print(type(A[0][keys_list[1]]))

