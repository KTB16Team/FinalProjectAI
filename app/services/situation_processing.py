# #%%
# import sys
# import os
# import pickle
# import json
# import ast
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# import openai
# import asyncio

# #%%
# # OPENAI_API_KEY
# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# client = openai.OpenAI(
#     api_key = OPENAI_API_KEY,
# )
# #%%
# async def summarize_and_evaluate_situation(ref_text: str) -> str:
#     llm = ChatOpenAI(model='gpt-4o', temperature=0.3)

#     # Combined prompt template for summarization and evaluation
#     combined_textbook = '''
#     original text : {ref_text}
#     You are an evaluator that performs both a summary of the situation and an objective analysis of each key event.
    
#     1. situation_summary:
#        - Summarize the situation provided by the user.
#        - Replace the user with A, the person they are speaking to with B, and other individuals with C, D, E, etc.
#        - Ensure that significant events involving each speaker are not omitted in the summary.
#        - Provide an objective and neutral summary.
       
#     2. situation evaluation:
#        For each case extracted from the summarized situation, focus on objective events, excluding attitudes or emotions.
       
#        Each situation case should follow this format:
#        - situation_case1, situation_case2, ... : Key situation evaluation cases extracted from the summary.
#            - event : A brief description of the event
#            - participants : Key participants in the event.
#            - result : The outcome or result of the event
#            - time_frame : “Time frame” refers to the chronological order of events based on their context. For example, 
#                          if the time frame for “situation_case1” is 1 and for “situation_case2” is 3, this indicates the sequential position of each event. 
#                          The sequence is arranged according to the cause-and-effect relationship or the timeline in which each case occurs.
#            - score : The score of the situation, ranging from 0 to 1 (0 being least important, 1 being most important)
    
#     Output format:
#     [
#         {{"situation_summary": "", 
#           "situation_case1": {{"event": "", "participants": "", "result": "", "time_frame": "", "score": 0.5}},
#           "situation_case2": {{"event": "", "participants": "", "result": "", "time_frame": "", "score": 0.5}},
#           ...
#         }}
#     ]
#     '''

#     # Generate the prompt with the provided ref_text
#     prompt = ChatPromptTemplate.from_template(combined_textbook)

#     # Chain connection
#     chain = prompt | llm | StrOutputParser()

#     # Generate response
#     response = chain.invoke({
#         'ref_text': ref_text
#     })

#     # Clean and parse JSON response
#     print(response)
#     cleaned_data = response.replace('json', '').replace('```', '').strip()
#     situations = json.loads(cleaned_data)

#     return situations

# async def behavior_evaluation(ref_text: str, situation_summary: str = None) -> str:
#     llm = ChatOpenAI(model='gpt-4o', temperature=0.3)

#     # Behavior evaluation prompt template with optional situation_summary
#     behavior_textbook = '''
#     original text : {ref_text}
#     {situation_summary_text}
#     You are an evaluator who analyzes behaviors and attitudes expressed in the provided situation summary.
#     Evaluate key behaviors, attitudes, impacts, and emotions based on the summary.
    
#     Each behavior case should follow this format:
#     - behavior_case1, behavior_case2, ... : Key behavior evaluation cases extracted from the summary.
#        - attitude : The attitude of the speaker
#        - behavior : The primary behavior or action performed by the speaker
#        - impact : The significant impact of the behavior
#        - emotion : The emotion expressed by the speaker during the interaction
#        - score : The score of the behavior 0 to 1 (0 being least important, 1 being most important)
    
#     Output format:
#     [
#         {{"behavior_case1": {{"attitude": "", "behavior": "", "impact": "", "emotion": "","score": 0.5}}},
#           "behavior_case2": {{"attitude": "", "behavior": "", "impact": "", "emotion": "","score": 0.5}},
#           ...
#         }}
#     ]
#     '''

#     # Insert situation_summary if provided
#     situation_summary_text = f"summarized situation : {situation_summary}" if situation_summary else ""
#     prompt = ChatPromptTemplate.from_template(behavior_textbook)

#     # Chain connection
#     chain = prompt | llm | StrOutputParser()

#     # Generate response
#     response = chain.invoke({
#         'ref_text': ref_text,
#         'situation_summary_text': situation_summary_text
#     })

#     # Clean and parse JSON response
#     cleaned_data = response.replace('json', '').replace('```', '').strip()
#     behaviors = json.loads(cleaned_data)

#     return behaviors

# async def emotion_evaluation(ref_text: str, situation_summary: str = None) -> str:
#     llm = ChatOpenAI(model='gpt-4o', temperature=0.3)

#     # Emotion evaluation prompt template
#     emotion_textbook = '''
#     Analyze the emotions in each dialogue entry below. For each entry, identify the speaker, calculate the emotion scores, and determine whether each emotion was given to the other person or received from them. Each detected emotion should be scored on a scale from 0 to 1, where 0 means no emotion and 1 means a very strong emotion.

#     Dialogue:
#     {dialogue_text}
#     {situation_summary_text}
#     Instructions:
#     1. For each dialogue entry, detect the emotions conveyed in the text.
#     2. Identify each emotion as either given to the other person or received from them.
#     3. Assign a score between 0 and 1 for each detected emotion in both categories.
#     4. Record the speaker (A or B) and the dialogue entry number.

#     Output format (JSON):
#     [
#         {{
#             "entry": 1,
#             "speaker": "A",
#             "given_emotions": {{
#                 "joy": 0.8,
#                 "trust": 0.0,
#                 "fear": 0.6,
#                 "surprise": 0.0,
#                 "sadness": 0.0,
#                 "disgust": 0.0,
#                 "anger": 0.0,
#                 "anticipation": 0.5
#             }},
#             "received_emotions": {{
#                 "joy": 0.8,
#                 "trust": 0.2,
#                 "fear": 0.0,
#                 "surprise": 0.0,
#                 "sadness": 0.0,
#                 "disgust": 0.0,
#                 "anger": 0.0,
#                 "anticipation": 0.0
#             }}
#         }},
#         {{
#             "entry": 2,
#             "speaker": "B",
#             "given_emotions": {{
#                 "joy": 0.6,
#                 "trust": 0.8,
#                 "fear": 0.0,
#                 "surprise": 0.0,
#                 "sadness": 0.0,
#                 "disgust": 0.0,
#                 "anger": 0.0,
#                 "anticipation": 0.3
#             }},
#             "received_emotions": {{
#                 "joy": 0.0,
#                 "trust": 0.0,
#                 "fear": 0.0,
#                 "surprise": 0.0,
#                 "sadness": 0.0,
#                 "disgust": 0.0,
#                 "anger": 0.0,
#                 "anticipation": 0.0
#             }}
#         }},
#         {{
#             "entry": 3,
#             "speaker": "A",
#             "given_emotions": {{
#                 "joy": 0.5,
#                 "trust": 0.9,
#                 "fear": 0.0,
#                 "surprise": 0.0,
#                 "sadness": 0.0,
#                 "disgust": 0.0,
#                 "anger": 0.0,
#                 "anticipation": 0.4
#             }},
#             "received_emotions": {{
#                 "joy": 0.5,
#                 "trust": 0.9,
#                 "fear": 0.0,
#                 "surprise": 0.0,
#                 "sadness": 0.0,
#                 "disgust": 0.0,
#                 "anger": 0.0,
#                 "anticipation": 0.0
#             }}
#         }}
#     ]
#     '''

#     # Prompt creation
#     situation_summary_text = f"summarized situation : {situation_summary}" if situation_summary else ""
#     prompt = ChatPromptTemplate.from_template(emotion_textbook)

#     # Chain connection
#     chain = prompt | llm | StrOutputParser()

#     # Generate response
#     response = chain.invoke({
#         'dialogue_text': ref_text,
#         'situation_summary_text': situation_summary_text
#     })

#     # Clean and parse JSON response
#     cleaned_data = response.replace('json', '').replace('```', '').strip()
#     emotions = json.loads(cleaned_data)

#     return emotions
# #%%

# test_Case1 = '''
# A: 또 30분이나 늦었네. 우리가 몇 시에 만나기로 했는지 기억해?

# B: 차가 막혀서 어쩔 수 없었어. 미리 연락했잖아. 정말 빨리 오려고 했다고.

# A: 그거야 그렇지. 근데 벌써 몇 번째야? 매번 이렇게 기다리게 하면, 나도 지치는 거 몰라?

# B: 미안하다고 했잖아. 일부러 늦은 것도 아니고… 상황이 어쩔 수 없었다니까.

# A: ‘어쩔 수 없었다’는 말 이제 그만 좀 듣고 싶어. 너는 항상 변명만 늘어놓고, 내가 매번 이해해야 한다고만 생각하는 거 같아.

# B: 아니야, 진짜 사정이 있었으니까 그런 거잖아. 이해해줄 수 없는 거야?

# A: 그럼, 네가 계속 이렇게 반복해서 늦을 때마다 내가 이해만 해야 하는 거야? 네가 얼마나 기다리게 하는 건지, 그동안 내가 어떻게 느끼는지 알기나 해?

# B: 나도 불편한데… 진짜 상황이 그랬으니까 어쩔 수 없다고 이해해줬으면 좋겠어. 기다리는 게 싫으면 딱히 억지로 기다릴 필요도 없었잖아.

# A: 억지로 기다렸다고 생각해? 그게 무슨 말이야? 너를 만나고 싶어서, 시간 맞춰 나오고 기다린 건데… 그게 억지라니. 그러면 차라리 네가 시간을 더 맞추려고 노력했어야지.

# B: 무슨 소리야, 나도 노력했어. 매번 늦고 싶은 게 아니라고. 근데 내가 뭘 할 수 있었겠어? 차가 막혀서 그랬는데 내가 어떻게 했어야 했는데?

# A: 시간 약속이 중요하다면, 애초에 일찍 출발하거나, 방법을 미리 생각해봤겠지. 언제나 ‘어쩔 수 없었다’는 말만 하니까 신뢰가 떨어지는 거잖아.

# B: 그래, 내가 매번 늦었던 건 아니잖아. 오늘은 상황이 안 좋았던 거지. 그리고 그런 말을 듣는 것도 나로선 부담스러워. 네가 날 전혀 이해 안 해주는 것 같아서…

# A: 네가 그러는 것처럼 나도 사정이 있겠지. 근데 내가 항상 네 입장을 이해해야 해? 나도 네가 기다리는 거 신경 써줬으면 좋겠는데 그게 그렇게 힘든 거야?

# B: 너도 내가 왜 늦었는지 이해해줘야지. 내 입장도 생각해주고, 상황이 어쩔 수 없는 건 받아들일 수 있는 거잖아. 계속 이렇게 불만을 쌓아가는 게 너한테도 좋은 건 아니라고 생각해.

# A: 난 네가 늦는 거보다 항상 이유를 찾아내서 자기 입장만 고집하는 게 더 답답해. 그럴 때마다 내가 무슨 생각을 하는지 네가 전혀 관심 없어 보여.

# B: 그러면 매번 무조건 미안하다고만 할 수는 없잖아. 나도 진짜 노력을 하고 있는데, 네가 자꾸 이 문제로 날 몰아붙이면 솔직히 화도 나고 힘들어.

# A: 이해 못할 것도 없겠지, 근데 네가 나한테 더 관심을 두고 노력했다면 이런 일이 없었을 거야. 우리는 약속할 때마다 반복되는 일이니까 그만큼 중요하게 생각해줬으면 좋겠어.

# B: 자꾸 이렇게 나한테만 불만을 쌓아놓으면 그게 정말 좋은 관계를 유지할 수 있는 방법이라고 생각해?

# A: 난 노력하라고 계속 말하는 거야. 그렇지 않고서는 넌 더 이해 못할 거 같아서… 이런 상황이 계속 반복되면 결국 우리 둘 다 지칠 거야.

# B: 정말 내가 널 이해 못 한다는 말밖에 안 들리네. 난 이게 내 잘못인 줄 알고 미안해하려고 해도 네가 그럴수록 더 부담돼.

# A: 이해 못하는 게 아니잖아, 그냥 내가 말한 것처럼 실수하고 있는 걸 알고 그 부분을 조심하자는 건데… 그런 게 어렵다고 생각해?

# B: 나도 최대한 네 입장을 생각하겠지만, 그럼 너도 내 상황을 한 번만이라도 이해해주는 게 그렇게 어려워?

# A: 그럼 네 말대로 무조건 이해해줄게. 네가 시간을 아무리 늦어도 다 이해하고 받아들이면 되겠네. 그래, 이렇게 하면 해결이 되긴 할 거야.

# B: 진짜로 그렇게 생각하는 거야? 그냥 네가 원하는 대로 난 이해만 하고 넘어가야 한다는 거야?

# A: 뭐, 그게 서로를 위해서라면 할 수도 있지. 문제는 네가 먼저 그렇게 하려고 하지 않았다는 거잖아. 그래도 난 널 배려하려고 노력했어.

# B: 배려는 양쪽이 다 해야 하는 거야. 내 입장에서도 너의 기분을 생각하려고 했는데, 네가 내 마음을 이해해주지 않으니까 힘들다고 느껴지는 거잖아.

# A: 그렇다면 난 매번 네가 하는 걸 받아들여야 하고, 너는 그대로 남아도 된다는 거네? 그게 무슨 관계야?

# B: 너도 참… 그렇게 모든 게 네 말대로만 가야 하는 거야? 난 노력하려는 것 자체를 말하는 건데.

# A: 그러면 내 말대로 안 할 거야? 어떻게 해야 네가 이해할 건데?

# B: 그건 서로 이해하고 맞춰가는 거야. 계속 서로가 받아들일 수 있게… 말로만 이해하라는 건 다 의미 없어져.
# A: 그럼 네 말대로 난 그냥 너한테 맞추고, 계속 이해만 해야 하는 거네? 그래, 그렇게 하면 되겠네. 네가 늦고, 매번 변명하고, 난 그냥 받아들이고… 이게 다야?

# B: 왜 그렇게 비꼬는 거야? 나도 이 상황이 힘들다고. 내 입장은 생각 안 하고 무조건 내 잘못만 강조하는 거 정말 지쳤어.

# A: 지쳤다고? 나는 안 지쳤겠어? 매번 같은 이유로 기다리고, 화내면 네 입장 생각해줘야 하고, 결국 내가 이해하라는 거잖아. 이게 반복되니까 더 힘들어.

# B: 그래서 너는 모든 잘못이 다 내 탓이라고 생각하는 거야? 그럼 이제 뭐 어쩌자는 거지? 계속 이 얘기만 하고 있을 거야?

# A: 내가 내 입장을 말하면 왜 매번 네가 억울해 하는지 모르겠어. 그렇게 힘들면 처음부터 시간을 맞췄어야지. 애초에 약속 시간 안 지키는 게 문제인 거잖아.

# B: 그럼 넌 한 번이라도 내가 왜 늦는지 이해하려고 했어? 차가 막혔고, 진짜 어쩔 수 없는 상황도 있었는데… 날 한 번이라도 이해하려고 노력은 했어?

# A: 이해해주려고 했지. 근데 너는 왜 매번 똑같은 핑계야? 차가 막혔다는 변명으로 계속 반복되는 게 너는 이해가 돼?

# B: 알겠어, 알겠다고. 그러면 내가 항상 틀렸고, 네가 옳다는 말이야? 난 진짜 지쳤어, 더 이상 말하기도 싫어.

# A: 그래, 그럼 더 얘기 안 할게. 어차피 얘기해봤자 네가 듣지 않는 것 같으니까.

# B: 어, 나도 그래. 네 말 다 들어봤으니까… 이제 뭐 더 말할 것도 없겠네.

# A: 네가 이해하든 말든 상관없어. 이런 식으로 하면 우리 그냥 끝나는 거야. 이 얘기를 계속해도 달라질 게 없으니까.

# B: 좋아, 그럼 여기서 끝내. 나도 이제는 진짜 이게 맞는 건가 싶어. 너도 집에 가.

# A: 좋아, 나도 집에 갈게. 이젠 할 말도 없어.

# B: 그래, 가라.

# A: 어차피 너도 나도 해결할 생각 없으니까, 그냥 끝내자.

# B: 그렇게 말하고 싶으면 그렇게 하자고.

# A: 그럼, 나 간다.

# B: 어, 나도 갈 거니까 더 이상 말하지 마.'''

# test_Case2 = '''
# 최동석 : 내가 뭐 티를 냈네, 인스타에 저격을 했네, 그런 얘기하지 마.
# 박지윤 : 커뮤니티에 니 인스타 캡처가 올라와서 사람들이 댓글을 달고, 그게 내 귀에 들어오니까 하는 말이야.
# 최동석 : 니가 떠들고 다닌 사람이 15명이 넘어.
# 박지윤 : 그 사람들이 전 국민이 알게 올렸어? 커뮤니티에 올라오게 올렸냐고?
# 최동석 : 그러니까 그런 얘기하지 말고.
# 박지윤 : 사적인 대화랑 공적인 공표랑 같냐고. 뭐 팩트 아닌 것도 짚어봐?
# 최동석 : 자 들어! 그 후배가 저번에 그 카페에서 너 (남자 만나는 거) 봤던 애야.
# 박지윤 : 그래.
# 최동석 : 근데 걔가 캡처를 해서 이렇게 보낸 거야. "남자 만나고 있는데 괜찮아요? 형?" 하면서.
# 박지윤 : 그러니까 그 사람 생각이 이상한 거 아니야. 
# 최동석 : 아니 근데 극단적으로.
# 박지윤 : 내가 그 남자랑 데이트를 했어? 거래처 직원이 퇴사하는데 고맙다고 인사한다 해서 커피 한 잔 마신 걸 가지고 남자 만나고 있다고? 그리고 거래처 오픈식 모임 사진에는 나만 있어? 어? 내가 혼자 놀러 다니고 거기서 내가 술을 마셨어? 밤이야? 
# 최동석 : 걔(후배) 입장에서 니가 노는 것처럼 보일 수 있어. 그 얘기를 하는 거야.
# 박지윤 : 그건 거기가 내 거래처인지 거래처 대표님 초대를 받아서 갔는지 앞뒤 상황을 모르는 후배가 넘겨짚은 거 아니야. 그러면 '공구하는 거래처 오픈식인데 갔나 보다' 그 후배한테 그렇게 얘기 못해?
# 최동석 : 내가 그래서 걔한테 한마디도 안 했어.
# 박지윤 : 한마디도 안 하니까 이상하게 생각하지. 어? 내가 너한테 실드를 쳐달래? 뭘 하래? 
# 최동석 : 들어봐. 그리고. 
# 박지윤 : 아니. (이건) 후배의 생각을 전하는 게 아니야. 니가 그 생각을 나한테 전하는 건 '나도 후배 말이 맞다'라고 하는 건데 
# 최동석 : 나도 열이 받아
# 박지윤 : 뭐가 열이 받는데?
# 최동석 : 그 자리에 안 가면 안 돼?
# 박지윤: 안 가면 안 되지.
# 최동석 : B 학비를 앞으로 내가 책임지겠다고 했어. 네가 일부 납부한 거 알고 난 다음이야. 근데 니가 A 학비 없다고 징징댔어.
# 박지윤 : 징징댄 거 아니고요. 본인이 "공구로 2억 벌었어. 공구가 개 쉬운데 이걸로 개생색냈냐? 그리고 나는 너한테 돈 안 줄 거다" 했잖아. 그래서 (내가) "그럼 돈이 있으시면 B 학비를 내가 냈으니 A 학비를 기간 내에 내십시오"라고 했지.
# 박지윤 : 그런데 니가 갑자기 나한테 800만 원을 보냈어. 그래서 내가 "800만 원이 무슨 돈이야? 나 안 받겠다" 했더니, (니가) "학비 달라 해놓고 안 받아?"라고 했잖아. 그래서 나는 "학비가 800이 아니며 3,000 가까이 된다"고 했지. 
# 최동석 : 다 떠나서, 다 떠나서, 학비 아니라고 치자. 돈을 줬어. 그러면 니가 생활비로 쓰든 학원비로 쓰면 되잖아. 
# 박지윤 : 나는 학비를 달라고 했지. 너 형편 될 때. 내가 생활비 달래? A 학비 내라는 거잖아.
# 최동석 : 그런데 왜 생활비 달라는 이야기를 계속하냐, 이거야?
# 박지윤 : 생활비 달라고 하는 게 아니야. 내가 돌아다니는 게 싫고, 거래처 만나는 게 싫고, 거래처 오픈식에 가는 게 싫고, 바깥 생활을 하는 게 싫으면, 아예 집구석에 있는 그림을 만들든가! 아니면, 기사 쓰고 아줌마 쓰고 내가 해결하고 나갈 수 있게 하든가!
# 최동석 : 그래서 하겠다잖아. 지금부터.
# 박지윤 : 지금부터 필요 없어. 왜 지난 시간 힘든 세월 다 지나갔는데…
# ④ 폐부
# 박지윤 : 그리고 내가 너가 800, 결혼기념일에 1,000? 적선하듯이 던져준 돈 왜 안 받았는지 알아?
# 최동석 : 내가 적선하려고 너한테 줬냐? 나 결혼 기념 상관없이 보낸거야.
# 박지윤 : 나는 내 자금 사정, 거래처 입금이 언제든, 내 월급이 얼마든, 상관없이 기간 내에 애들 학비 다 내고 살았어. 내 세금 못내서 카드로 할부로 내면서도. 
# 박지윤: 너는 니 세금 다 내고, 니 차 다 사고, 남는 돈 너 형편 될 때 찔끔찔끔 나 줬잖아. 그러면서 평생 이 집에 살면서 기여한 돈이 솔직히 내가 1원이라도 더 많은 게 팩트인데. "니 엄마 빚 갚아준 돈 3,000만 원 내놔"라고 했지.
# 최동석 : 내가 언제 내놓으라 그랬어?
# 박지윤 : 내놓으라고 했어. 분명히
# 최동석 : 네가 치사해서 준다고 그랬지.
# 박지윤 : "니 엄마 빚 갚아준 돈 3,000 갚아라"고 했어. 그래서 내가 그 돈 갚는다고 했더니, "XX한테 뜯긴 돈 갚아주는 사람 없다"면서 생색을 내고 마누라 폐부를 찢어놨지.
# 박지윤 : 어? 니 엄마 빚 갚아준 돈, (그 돈) 갚으라는 남편한테 누가 돈을 받냐. 800? 1000? 야, 애 학비도 기간 내에 못 내면서, 너 형편 될 때 던져주는 돈 필요 없다고. 할 거면 니 역할을 정확히 반을 하라고.
# 최동석 : 그래서 앞으로 반한다니까.
# 박지윤 : 앞으로?
# 최동석 : B 학비 내가 낸다고
# 박지윤 : 아니 그러니까 오지 않은 얘기하지 마. 그때 하고나 생색 내. 지금까지 못 한 거는 인정하고 그때 낸 다음에 얘기해.
# 최동석 : 내가 (학비) 못한 건, 어차피 나는 못한다고 얘기했어. 그래서 국제학교 보내는 거 반대했고. 보낸 거 너야.
# 박지윤 : 그래서 중간에 서울 가자니까.
# 최동석 : 니가 다 책임지겠습니다. 맹세하고 그러고 다닌 게 제주도야.
# 박지윤 : 다 책임지게 됐습니다. 맹세해 왔으면 마누라 일하게 놔둬야지. 거래처 직원을 만나네 마네. 몇 시에 출도착을 하네 마네.
# 최동석 : 너는 중요한 게 말이 맨날 바뀌어. 그 남자가 거래처 직원이라고 했어.
# 박지윤: 거래처 직원.
# 최동석 : 그전에는 자기가 거래하던 명품 회사 명품 매장.
# 박지윤 : 거래처 직원! S사 직원. 명품 회사 직원. 나 거래처 S사 직원이라고 분명히 얘기했어. 그래서 그 회사 옆 커피숍에서 만났다고. 
# 박지윤 : 이거 봐. 너는 내 말꼬리 잡으면서 그게 팩트야? 너 그럼 담당자한테 전화해봐라. 그 사람이 누구고, 퇴사했는지 안 했는지? 그리고 니 후배한테 실물 대조해 봐. 이런 대화를 왜 해야 돼? 그러니까 니가 병이라는 거야.
# 최동석 : 그게 남자(직원)이라 문제가 아니라, 그게 니 업무와 관련이 됐냐 안 됐냐의 문제야!
# 박지윤 : 내 업무야. 내 일과 관련이 된거야.
# 최동석 : (그럼) 바자회는 네 일이야 ?
# 박지윤 : 내 일이야!
# 최동석 : ㅈ까!
# 박지윤 : 내가 너 출장에서 누구 만나는지 검색해? 내가 거래처 여직원과 동석하면 만나지 마라고 해? 아니잖아. 애초에 그 구조를 만들어놓고.
# 최동석 : 난 그런 일이 없으니까.
# 박지윤 : 그런 일이 있는지 없는지 어떻게 알아? 너는 사사건건 내 친구 스토리 염탐하고 있으니까 그걸 알지. 어? 
# 박지윤 : 놀러 간 게 분명히 아니야. 거기 있는 사람들 다 비즈니스 오픈식 축하하러 온 사람들이야. 그런데 그걸 왜곡된 시선으로 보는 니가 병이라는 거야. 그걸 왜 딴지를 걸어? 
# 박지윤 : 니 말대로 학비를 책임지고. 생활비 그건 딱 반만 해. 그동안 니 차도 내가 했고, 애들 학비도 내가 했고, 애들 학원비도 내가 한 거 맞잖아. 
# 최동석 : 그 차 얘기 잘 꺼냈어. 내가 내가 차 왜 샀는데? 내가 가오부리려고 샀니? 어? 네가 (내) 차값 내는 거 부담스럽다고 해서 빨리 팔고 해치우고.
# 박지윤 : 빨리 해치우려는 사람이 차 나오기까지 그 차 빨리 할부 해지하거나 넘기라는 얘기 안 했잖아. 너는 내가 한 달 비용을 더 내든 말든, 그거 상관이 없는 거잖아. 
# 박지윤 : 그럼 새 차 언제 출고될 거니까, 그 차 정리하라고 했어? 아니! 나는 니가 차를 뽑은 사실을 다른 사람들이 말해줘서 했어. 
# 최동석 : 차 정리하라고 얘기했잖아?
# 박지윤 : 정리하라고 얘기 안 했어. 니가 "(내) 차 정리하면 나는 라이드를 못하니까 니가 다 라이드를 해라"는 협박식으로 얘기했지. 나는 니 차가 출고된 사실은 다른 사람을 통해서 들었지. 
# 박지윤 : 그때 내가 실망이 들더라. 내가  한 달에 200만 원 할부금 더 내도 이런 거는 아무 상관이 없구나. 어찌 됐든 너가 움직이는 이동 수단 하나 니 능력으로 못 했었잖아. 그러면서 왜 내 생활에..
# 최동석 : 내가 안 한 게 아니잖아.
# 박지윤 :  내 비즈니스에 감 놔라 배 놔라를 해? 내가 아줌마건 뭐건 다 구해 놓고 일할 테니까 서울 가서 직장 다니라고 하는데, 한시도 아이들과 떨어져 있을 수 없다고 회사를 그만둔 게 너야.
# 최동석 : 자, 다 떠나서 네가 욕지거리하는 거 걸려가지고 가정이 파탄 났어. 내가 왜 그것 때문에 이혼 꼬리표를 달아야 돼? 내가 왜 애를 만나는 데 불편함을 겪어야 되냐?
# 박지윤 : 아니! 내가 욕지거리 한 거는, 난 친구들하고 남편 욕 X발 X발 할 수 있다고 생각해. 
# 박지윤 : 근데 너는 그게 가정 파탄의 기준이었지. 니가 못 살겠다 했고, (그래서) 내가 이혼하자고 했지. 근데 니가 차를 돌려와서 무릎 꿇고 이혼하지 말자고 한 게 너야.
# 최동석 : 니가 제정신이 아닌게 뭔지 알아? 욕 좀 한 거 가지고 이 난리를 피운다고? 어? 욕 좀 한 거 가지고.
# 박지윤 : 그럼 욕이지 아니야?
# 최동석 : 욕 좀이야? 그게 단순하게 그냥 남 뒷담화치다 걸린 그 수준의 지금 상황이야. 이게?
# 박지윤 : 너는 아무것도 안 하고 가정생활에 충실하고 너무나 잘했는데 내가 욕지거리를 했어? 너 맨날 짜증 냈잖아.
# 최동석 : 내가 그래서 누구한테 다 떠벌리고 다녔냐고?
# 박지윤 : 내가 뭐 떠벌리고 다녔어? 나는 내 친구들이 있는 단톡방에 얘기했어.
# 최동석 : 니가 사고 치고, 앵커 잘리고, 제주도 쫓겨 왔고, 재취업도 못 하게 만들었고, 그 상황에서 우울증이 안 와? 
# 최동석 : 그래서 병원 갔어. 근데 네가 나한테 뭐라고 그랬어? 정신병자라고 그랬지. 
# 박지윤 : 내가 너 정신병자라고 한 거는 너의 의처증 때문에 정신병자라고 한거야.
# 최동석 : 의처증이고 지X이고 너는 분명히 다른 남자 만나고 다녔고.
# 박지윤 : 다른 남자 만나고 다닌 적 없어! 너의 망상이지. 증거 있냐고!
# 최동석 : 망상 아니야 ! 팩트야!
# 박지윤 : 팩트만 대. 소송 걸어. 근데 왜 안 살겠다고 하는 나한테 괴롭히면서 지옥에서 살라고 하냐고.
# 최동석 : X발, 나도 이혼하고 싶다고 진짜.
# 박지윤 : 그러면 이혼하자고. 욕지거리하고 나한테 맨날 소리 지르고.
# 최동석 : 그러니까 너도 거슬리게 행동하지 말고. 그냥 닥치고 살아.
# 박지윤 : 뭘 닥치고 살아. 내가 거슬리게 뭘 행동을 했는데? 내 일 하는 거 가지고 거슬린다고 하는 니가 정신병자인거야. 어? 
# 박지윤 : 내가 가만히 집구석에 있게 만들었어? 아니잖아. 애들 1명 학비를 떠나서 2명 학비에 생활비, 다 내 차에 다 가능하게 만들어 놓고 집구석에 있으라는 소리를 해?
# 최동석 : 누나! 누나 이상형이에요. 누나 같은 사람 소개시켜주세요. 누나 설레요. 보고 싶어요.
# 박지윤 : 그럼 거기다가 "씨XX아 닥쳐"라고 해?
# 최동석 : 닥치라고 해. 제발
# 박지윤 : 그러니까 너는 사람 사이에 예의도 없고.
# 최동석 : 서귀포에 있던 애가 "누나 데리러 갈게요" 그러면, 니가 그런 시그널이 있었으면 오지 말라고 했어야지.
# 박지윤 : 나는 시그널인지도 몰랐고요. 
# 최동석 : 그게 몰랐다고 하는 게 말이 안 되는 거야.
# 박지윤 : 왜 말이 안 돼? 너는 모든 남녀 관계를 섹스로만 보고, 너 망상으로 보고, 의심으로 보니까, 모든 게 다 시그널이지. 거래처 직원 만나서 커피 마셔도 남자 만나는 거. 거래처 오픈식 사진에 남자가 없었으면 너 분노 버튼이 안 눌려졌을까?
# 박지윤 : 그리고 같이 방송하고 친하게 지냈던 후배가 "누나 마침 제주도 오셨다면서요? 제가 데리러 갈까요?" 그러면 차 한번 얻어 탈 수도 있지. 아니 아니 난 그렇게 생각하는데.
# 최동석 : 아니! 
# 박지윤 : 아니, 난 그렇게 생각하는데!
# 박지윤 : 5년 전에 이 일로 싸울 때 마다, "넌 그 XX 차를 탔잖아". 싸울 때마다, "넌 그 XX 차를 탔잖아". 지금 6개월째 그 이야기를 들어야 되면, 우리 결혼 생활은 파탄인데?
# 최동석 : 그냥 사과를 하라고
# 박지윤 : 뭔 사과를 해? 너와 나의 기준이 다른데 
# 최동석 : 그냥 입장 바꿔놓고. 닥쳐 조용히 해. X발
# 박지윤: 내가 무릎 꿇고 너한테 안 빌었어? "그래, 내가 다 잘못했다. 그러니까 제발 그만 살자" 했어. 안 했어? 그러니까
# 최동석 : '그만 살자'가 아니라고. '잘못했으니까 너한테 선택을 맡길게'가 정답이라고.
# 박지윤 : 왜 내가 너한테 선택을 맡겨야 돼? 내 인생이 네 거야? 어? 나는 개인 의지도 없고, 너가 시키는 대로 해야 되는 사람이야?
# 최동석 : 아는 여자애가 있어. 근데 얘가 "오빠, 오빠 이상형이에요. 오빠 같은 남자 만났으면 좋았을 텐데"라고 해.
# 박지윤 : 그런 사람 많아. 주변에 최 아나운서가 이상형이라는 사람. 이렇게 X발 X발 욕하는지 모르고.
# 최동석 : 들어봐. 내가 개인적인 상황이라고 했잖아. 개인적으로 아는 여자애가 그랬어. 어? 그런 상황에 뭐 "오빠를 보면 설레고요. 오빠 같은 남자도 너무 좋고요" 막 이런 애야. 만나고 싶어 해. 계속 사적으로 우리 술 마시자고 계속 연락이 와.
# 박지윤 : 그래서 사적으로 내가 같이 술을 마셨으면 문제가 됐겠지.
# 최동석 : 술 마셨잖아.
# 박지윤 : 술 안 마셨어. 공적으로 마신 거지.
# 최동석 : 단둘이 마셔요.
# 박지윤 : 단둘이 술을 마셨어? 내가? 그 언니가 그 사람 만나는데 같이 가자 해서 대낮에 식사 한번 한 게 다라고. 
# 최동석 : 그래서 그 얘기를 아무렇지 않게? 단둘이? 유부녀라는 사람이?
# 박지윤 : 이게 5년 전 일인데, 5년 전에 차 한 번 얻어 탄 걸 가지고, 내가 왜 우리 부부 싸움이 다른 문제로 싸울 때마다 너는 그 XX를 찾고 있잖아. 
# 박지윤 : 왜 내가 있지도 않은 내 감정을 왜곡당하는 이런 추궁을 당해야 하며, 내가 공적으로 만나는 거에도 거기에 남자만 껴 있으면 너는 분노 버튼이 일어나잖아.
# 최동석 : 자!
# 박지윤: 그러니까 나는 이런 게 다 싫다는거야. 너와 나는 가치관이 다른데 어떻게 이 긴 세월을 살아가? 내가 제발 우리는 다르니 무릎 꿇고 빌면서 진짜 제발 이혼만 하자 했지. 그랬더니 너는 "니 마음대로 하고 각자도생하면서 살자"고 했고. 
# 박지윤 : 근데 이게 니 마음대로야 ? 내가 출장 가서 거래처 오픈식에 갔고 다음 날 바자회에서 내 물건 팔아서 자금 마련도 하고 기부도 하겠다는데. 그 바자회를 지난 10여 년 동안 계속했던 건데..
# 박지윤 : 너는 "왜 (바자회를) 하냐"고 하지. 내가 너의 잣대로 하지 말아야 할 행동과 해야 할 행동을 구분받아야 돼? 하물며 그 자리에 있는 가정 주부들도 와 있었고, 아무도 뭐라 안해. 
# 박지윤 : 다들 '니가 이런 소리를 왜 듣고 살아야 되냐'며 미쳤다고 할거다. 나도 내가 어이가 없어. 나 진짜 지난 세월 동안 이런 걸로 눈치 보면서 사는 게 이해가 안 가.
# 최동석 : 자, 이거 하나만 대답해 봐. 그러면 여자애가 그렇게 껄떡껄떡 대고 있었어.
# 박지윤 : 그게 껄떡이라고 생각하는 니가 잘못한 거지. 진짜 존경하는 누나에게 "롤모델이고, 이상형이에요"라 할 수있지. 
# 박지윤 : 나 어제도 들었어. "아나운서님 같은 사람이 이상형"이라고. "아 그래요? 감사합니다" 그러고 마는 거지. 그럼 거기대 대고 "씨XX아! 나는 가정 있는 여자야!"이래?
# 최동석 : 지속적으로 걔가 그랬잖아. 
# 박지윤 : 나는 기억도 안 난다고요.
# 최동석 : 상당 기간 동안, 여러 차례 계속해서 만나고 있고.
# 박지윤 : 너는 그게 거슬리니까 횟수를 헤아리고 있었나 보지. 나는 상관이 없어서 나 먹고살기 바빠서.
# 최동석 : 입장 바꿔서 생각해 보라고. 내가 그 상황에서?
# 박지윤 : 아니! 그래서 입장을 바꿔도, 5년 전 일에 대해 내가 잘못하지도 않은 걸 가지고 내가 무릎 꿇고 사과를 해야겠냐고?
# 최동석 : 야! 
# 박지윤 : 껄떡대지 말고 집구석에 처박혀 있으라고 하니까 싸움이 되는 거 아니야. 
# 최동석 : 아파트 빨리 팔아 제발. 제발. 
# 박지윤 : 나 안 팔고 싶은 거 아니야! 나 팔 거야. 대출 이자도 나가고 있어서 나도 부담이야.
# 최동석 : 압구정 팔라고.
# 박지윤: 압구정 집을 왜 니 마음대로 팔려고 해? 내 명의인데. 압구정 집만 팔면 돼 ? 너 항상 이혼 얘기 나오면 압구정 집 팔라고 하지. 그러면 부동산이 압구정 집뿐만 아니라 부모님 사시는 집도 있는데, 부모님 집 건드리면?
# 최동석 : 팔라고 했잖아. 
# 박지윤 : 우리 엄마를 내쫓다는 둥 발작거렸잖아. 니가.
# 최동석 : 팔라고 했잖아 그래서. 
# 박지윤 : 니가 팔라고 해서 집 내놨는데, 너 뭐라 그랬어? "추운 겨울에 우리 엄마 집 보러 다니게 한다"고 했잖아, 니가
# 최동석 : 그러니까, 집 내놓으라고. 
# 박지윤 : 니가 내놓으라고 한 걸 내놨는데도, 왜 내가 시어머니 내쫓은 사람이 되야 하냐고?
# 최동석 : 안 내놨다며? 
# 박지윤: 안 내놨어. 그런데 "추운 겨울에 우리 엄마가 너 때문에 집 보러 다닌다"고 했어, 안 했어?
# 최동석 : 내놨네!
# 박지윤: 안 내놨다고. 난 분명히 아주버님한테 얘기했어. "내가 이래서 싸우고 힘들다. 그래서 집을 팔게 될 수도 있을 것 같다"고. 
# 박지윤 : 너 면전에다 하는 욕은 괜찮냐? 면전에다 씨XX 하는 건 괜찮아? 왜 X발이야? 말할 때마다 욕하지 말라고 했지.
# 최동석 : 내가 그 얘길 들었어. '아나운서님이 A를 때린다'는 소문이 있다.
# 박지윤 : 그건 OOO이 낸 이상한 헛소문이잖아.
# 최동석 : 디테일에 아주 디테일해. OOO 애 생일 파티에서
# 박지윤 : 내가 A를 그 차에서 끌어낸 것밖에 없고. 그 현장에서 다른 사람들도 봤어. 그러니까 너는 내가 아이를 훈육한 걸 가지고도 폭력 엄마로 매도하고. 
# 박지윤 : 그러면 너는 너는 애 앞에서 "네 엄마가 다른 남자한테 꼬리를 쳤어"라고 하는 건 훈육이야? 양육이야?
# 최동석 : 팩트지 
# 박지윤 : 그건 폭력이야. 정서적 폭력. 그러면 내가 다 A 앞에서 얘기할까? 니네 아빠가 나 겁탈하려고 했다. 성폭행하려고 했다. 
# 최동석 : 왜? 그건 부부끼리 그럴 수 있는 거야. 
# 박지윤 : 부부끼리도 성폭행이 성립이 돼. 
# 최동석 : 야! 그럼 A가 결혼해서 남자친구가 대갈통 때리면 참아야 되냐?
# 박지윤 : 야! 그럼 결혼해서 남자친구가 니 사진 인터넷에 뿌린다면 참아야 되냐? 남자친구가 성관계하는 사이라고 강제로 하려고 그러면 참아야 되냐? 너 연애할 때 그랬잖아 누구 조심하라. 이OO 조심하라. 일면식도 없는 사람한테 어? 
# 박지윤 : 기사만 나도 그러잖아. 니가 어떻게 했길래. 그리고 내가 친구 생일 간다고 그랬는데 그 생일에 왜 가냐고 했지? 그때 병인 걸 알아봤어야 하는데. 결국은 15년째 이러고 살잖아.
# 박지윤 : 거래처 직원을 왜 만났네? 거래처 직원이 남자면 동석을 하지 말라! 그러면 매니저는 남잔데 스케줄은 어떻게 가고, 남자 MC가 있는 프로그램에 남자 PD가 있는데 방송은 어떻게 하냐? 그거는 돈 벌어야 되니까 이를 악물고 참았어?
# 최동석 : 니가 니가 섹스를 안 하고 바람을 안 피웠다고 주장하지만, 그게 정서적 바람이라고. 
# 박지윤 : 정서적으로 바람을? 
# 최동석 : 이성이 너한테 호감을 보인 걸 즐긴 거야. 
# 박지윤 : 즐기지 않았고요. 저는 걔가 호감이 있다는 사실도 처음에 인지하지 못했고요. 나중에 그냥 느낌이 그런가 해서 찜찜해서 연락을 안 한 것 뿐이고요.
# 최동석 : 팔로우를 서로 끊을 정도로 그 상황은 왜 벌어지는 건데.
# 박지윤 : 나 팔로우 한 번 싹 정리했어. 나는 남편도 정리하는 마당에 왜 팔로우를 못 끊어? 그리고 걔가 나를 왜 끊었는지 내가 그럼 물어봐?  내가 절친하게 지냈던 OO이도 언팔하는 마당에, (라디오) 출연자였던 남자 하나 언팔을 못해. 
# 박지윤 : 아니, 서로 언팔을 했냐 안 했냐까지 추궁하면 이게 병이지. 왜 이런 대화를 듣고 있어야 되냐고? 하루가 멀다 하고, 왜 이 대화로 싸워야 하냐고? 
# 박지윤 : 그러니까 제발 이혼하자. 지금 이 상황이 정상이야? 엄마 아빠 싸우는 소리가 애들한테 안들려? 이게 자식을 위하는 거야? 애들이 얼마나 불안해하겠냐고.
# 최동석 : 처음에 처음에 그냥 미안하다고 무릎 꿇었으면 돼. 그럼 나도 더 이상 얘기 안 하고 이렇게 싸울 일도 없어.
# 박지윤 : 미안하다는 것도 어느 정도껏 사람을 몰아붙여야 미안한 감정이 드는 거지. 마누라 없는 사이에 전화기랑 핸드폰 다 뒤지고. 자고 있는 사람 발로 차서 깨워서, 매일 밤 고문하듯이 4년 전 문자를 읽어대는데. 거기서 어떻게 미안하다는 얘기가 나와?
# 박지윤 : 그러고 갑자기 안아달라고 그러지. 내가 분노해서 화분을 던졌더니 분노 조절 장애라면서 그걸 사진 찍고 있지. 사이코패스 같은 너한테 어떻게 내가 미안하다는 말이 나와? 사람을 적당히 몰아야 미안하지. 
# 박지윤 : 니가 저지른 정서적 폭력이 더 심한데, 어떻게 미안하다는 말이 나와. 오죽하면 사람이 진짜 안 되겠다 싶으니까 죽으려고 했겠어. 정말 이 굴레를 끊을 수가 없는 거니? 제발 그만하자. 나 죽을 것 같으니까 그만하자고. 제발 살아만 있게 하자. 
# 박지윤 : 이 가정은 깨져서는 안 되고, 매일 애들한테 큰 소리가 나는 불안한 상황을 만들면서. 내가 듣기 싫어서 귀 막으면 조롱하듯이 영상 찍고 있어. 나는 그런 행동에 소름이 끼쳐. 그러니까 그만하자고. 
# 2023년 10월 13일, 두 사람은 1시간 동안 감정을 토해냈다. 그리고 17일 뒤, 이혼소장이 접수됐다. '말싸움'이 아닌, '증거싸움'이 시작된 것.

# '''
# #%%
# # 파일 열고 읽기
# # with open("test_Case1.txt", "r") as file:
# #     content = file.read()  # 파일 전체 읽기
# #     print(content)

# content = test_Case
# #%%
# A = asyncio.run(summarize_and_evaluate_situation(content))
# B = asyncio.run(behavior_evaluation(content,A[0]['situation_summary']))
# C = asyncio.run(emotion_evaluation(content,A[0]['situation_summary']))
# #%%

# def async test:
#     A = await summarize_and_evaluate_situation(content)
#     return print(A)
# # print(A)
# keys_list = list(A[0].keys())
# print(A[0]['situation_summary'])
# print(keys_list)
# print(A[0][keys_list[1]])
# print(type(A[0][keys_list[1]]))

