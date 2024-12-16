# test_empathy_score.py

import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from transformers import AutoTokenizer, AutoModel
from app.services.empathy_model import DialogueEmpathyModel, DialogueDataset

def test_model_prediction(model_path):
    """공감 예측 테스트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 로드
    try:
        model = DialogueEmpathyModel(
            input_size=768,
            hidden_size=512,
            num_speakers=10,
            num_layers=1,
            dropout=0.5
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # BERT 모델 초기화
    if not hasattr(DialogueDataset, 'bert'):
        DialogueDataset.bert = AutoModel.from_pretrained('skt/kobert-base-v1')
        DialogueDataset.bert.eval()
        
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    
    # 테스트 케이스
    test_cases = [
        ("오늘 발표했는데 긴장해서 실수했어...", "헐 발표면 진짜 많이 긴장되었겠다. 실수는 누구나 할 수 있지! 담에 잘하면 되지. 너무 수고했다 진짜!"),
        ("새로운 취미를 시작해보고 싶은데 뭐가 좋을까?", "요즘 힘들어서 기분 전환이 필요했구나. 같이 찾아볼까?"),
        ("일이 너무 많아서 지쳐요.", "뭐 어떡해 해야지.")
    ]
    
    print("\n=== 공감 예측 테스트 ===")
    
    with torch.no_grad():
        for speaker1, speaker2 in test_cases:
            print(f"\nA: {speaker1}")
            print(f"B: {speaker2}")
            
            # 텍스트 인코딩
            inputs = tokenizer(
                speaker2,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # BERT 특성 추출
            outputs = DialogueDataset.bert(**inputs)
            hidden_states = outputs.last_hidden_state.mean(dim=1)
            
            # 예측
            utterance_vector = hidden_states.unsqueeze(0)
            speaker_ids = torch.tensor([[0]]).to(device)
            prediction = model(utterance_vector, speaker_ids)
            score = prediction.squeeze().item()
            
            print(f"예측된 공감 점수: {score:.2f}")
            
            # 공감 수준 판단
            if score <= 0.39:
                print("낮은 공감 수준")
            elif score <= 0.79:
                print("중간 공감 수준")
            else:
                print("높은 공감 수준")

def test_model_loading(model_path):
    """모델 로드 테스트"""
    print("\n=== 모델 로드 테스트 ===")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        model = DialogueEmpathyModel(
            input_size=768,
            hidden_size=512,
            num_speakers=10,
            num_layers=1,
            dropout=0.5
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("모델 로드 성공!")
        
        # 모델 구조 출력
        print("\n모델 구조:")
        print(model)
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n총 파라미터 수: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return False

if __name__ == "__main__":
    # 모델 경로 설정
    checkpoint_path = os.path.join(project_root, '/Users/alice.kim/Desktop/aa/checkpoints/best_model.pt')
    # 모델 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model file not found at {checkpoint_path}")
        print(f"Looking for model at: {checkpoint_path}")
    else:
        # 모델 로드 테스트
        if test_model_loading(checkpoint_path):
            # 예측 테스트
            test_model_prediction(checkpoint_path)