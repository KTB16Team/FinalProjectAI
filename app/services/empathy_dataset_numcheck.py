import json

def analyze_dialogue_dataset(data):
    total_dialogues = len(data['dialogues'])
    
    empathy_ranges = {
        'low': 0,    
        'medium': 0, 
        'high': 0    
    }
    
    for dialogue in data['dialogues']:
        utterances = dialogue['utterances']
        
        for utterance in utterances[1:]:  # 첫 발화 제외
            score = utterance['empathy_score']
            if score <= 0.3:
                empathy_ranges['low'] += 1
            elif score <= 0.7:
                empathy_ranges['medium'] += 1
            else:
                empathy_ranges['high'] += 1
    
    print(f"총 대화 수: {total_dialogues}개")
    print("\n공감 점수 분포:")
    print(f"낮음 (0.0-0.3): {empathy_ranges['low']}개")
    print(f"중간 (0.4-0.7): {empathy_ranges['medium']}개")
    print(f"높음 (0.8-1.0): {empathy_ranges['high']}개")
    
def load_and_analyze_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            analyze_dialogue_dataset(data)
    except Exception as e:
        print(f"Error: {e}")


load_and_analyze_dataset('/Users/alice.kim/Desktop/aa/Final/app/services/empathy_dataset.json')