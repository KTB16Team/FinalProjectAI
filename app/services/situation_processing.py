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
# 파일 열고 읽기
with open("test_Case1.txt", "r") as file:
    content = file.read()  # 파일 전체 읽기
    print(content)
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
print(type(A[0][keys_list[1]])

