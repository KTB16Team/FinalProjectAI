from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# 사용자 맞춤형 프로필 (변경 없음)
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

def gpt_mediation(system_message, user_message):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message['content']

@app.route("/mediate", methods=["POST"])
def mediate():
    data = request.json
    user_id = data.get("user_id")
    situation = data.get("situation")
    user_statement = data.get("user_statement")
    partner_statement = data.get("partner_statement")
    
    user_profile = user_profiles.get(user_id, {})
    
    context = ""
    if user_profile.get("coffee_with_opposite_gender") > 15:
        context += "The user is sensitive about their partner having coffee with friends of the opposite gender. "
    if user_profile.get("alcohol_with_opposite_gender") > 30:
        context += "The user is particularly sensitive about their partner drinking alcohol with friends of the opposite gender. "
    
    # 상황 요약
    system_message = "You are an expert relationship counselor tasked with summarizing complex relationship situations concisely and objectively."
    summary_prompt = f"Please summarize the following situation in 2-3 sentences, focusing on the key points of conflict: {situation}"
    situation_summary = gpt_mediation(system_message, summary_prompt)
    
    # 사용자 입장 분석
    system_message = "You are an empathetic relationship analyst. Your task is to analyze the user's statement, considering their emotional state and underlying concerns. Provide insights into their perspective without judgment."
    user_prompt = f"Context: {context}\nUser's statement: '{user_statement}'\nAnalyze the user's perspective, emotions, and concerns in 3-4 sentences."
    user_analysis = gpt_mediation(system_message, user_prompt)
    
    # 상대방 입장 분석
    system_message = "You are an impartial relationship analyst. Your task is to analyze the partner's statement, considering their emotional state and underlying concerns. Provide insights into their perspective without bias."
    partner_prompt = f"Context: {context}\nPartner's statement: '{partner_statement}'\nAnalyze the partner's perspective, emotions, and concerns in 3-4 sentences."
    partner_analysis = gpt_mediation(system_message, partner_prompt)
    
    # 과실 분석 및 결론
    system_message = "You are a fair and balanced relationship mediator. Your task is to analyze the situation, determine fault ratios if applicable, and provide a constructive conclusion that promotes understanding and resolution."
    fault_prompt = f"""
    Situation Summary: {situation_summary}
    User's Perspective: {user_analysis}
    Partner's Perspective: {partner_analysis}
    
    Based on the above information:
    1. Determine if assigning fault is appropriate in this situation. If not, explain why.
    2. If fault can be assigned, provide a ratio (e.g., 60:40) and brief explanation.
    3. Offer a constructive conclusion that acknowledges both perspectives and suggests a path forward for the couple.
    Limit your response to 5-6 sentences.
    """
    fault_analysis = gpt_mediation(system_message, fault_prompt)
    
    return jsonify({
        "situation_summary": situation_summary,
        "user_analysis": user_analysis,
        "partner_analysis": partner_analysis,
        "fault_analysis": fault_analysis
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)