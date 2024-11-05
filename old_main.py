from fastapi import FastAPI, Request
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# 사용자 맞춤형 프로필
user_profiles = {
    "userA": {
        "coffee_with_opposite_gender": 20,
        "alcohol_with_opposite_gender": 40
    },
    "userB": {
        "coffee_with_opposite_gender": 10,
        "alcohol_with_opposite_gender": 50
    }
}

async def gpt_mediation(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides relationship mediation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message['content']

@app.post("/mediate")
async def mediate(request: Request):
    # JSON 데이터로부터 사용자 정보 및 대화 추출
    data = await request.json()
    user_id = data.get("user_id")
    situation = data.get("situation")
    user_statement = data.get("user_statement")
    partner_statement = data.get("partner_statement")
    
    # 사용자 맞춤형 가중치 적용 (필요한 경우)
    user_profile = user_profiles.get(user_id, {})
    
    # 상황 맥락 설명
    context = ""
    if user_profile.get("coffee_with_opposite_gender") > 15:
        context += "The user is sensitive about their partner having coffee with friends of the opposite gender. "
    if user_profile.get("alcohol_with_opposite_gender") > 30:
        context += "The user is particularly sensitive about their partner drinking alcohol with friends of the opposite gender. "
    
    # 1. 상황 요약
    summary_prompt = f"Please summarize the following situation briefly: {situation}"
    situation_summary = await gpt_mediation(summary_prompt)
    
    # 2. 사용자 입장 분석
    user_prompt = f"Based on the context '{context}', analyze the user's statement: '{user_statement}'"
    user_analysis = await gpt_mediation(user_prompt)
    
    # 3. 상대방 입장 분석
    partner_prompt = f"Based on the context '{context}', analyze the partner's statement: '{partner_statement}'"
    partner_analysis = await gpt_mediation(partner_prompt)
    
    # 4. 최종 과실 비율 및 결론
    fault_prompt = f"Given the situation '{situation_summary}', the user's statement '{user_statement}', and the partner's statement '{partner_statement}', please determine the fault ratio and provide a conclusion."
    fault_analysis = await gpt_mediation(fault_prompt)
    
    # 결과 반환
    return {
        "situation_summary": situation_summary,
        "user_analysis": user_analysis,
        "partner_analysis": partner_analysis,
        "fault_analysis": fault_analysis
    }