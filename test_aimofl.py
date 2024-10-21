import requests
import json

def test_mediation():
    url = "http://localhost:5002/mediate"
    
    payload = {
        "user_id": "userA",
        "situation": "My girlfriend wants to have dinner with her male coworker to discuss a work project.",
        "user_statement": "I don't feel comfortable with her having dinner alone with another guy. It feels disrespectful to our relationship.",
        "partner_statement": "It's just a work dinner. I need to discuss this project, and it's important for my career. You should trust me more."
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        print("Situation Summary:", result['situation_summary'])
        print("\nUser Analysis:", result['user_analysis'])
        print("\nPartner Analysis:", result['partner_analysis'])
        print("\nFault Analysis:", result['fault_analysis'])
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_mediation()