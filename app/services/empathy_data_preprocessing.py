import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer
from empathy_score import DialogueEmpathyModel, EmpathyLoss,train_model, eval_model
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoModel
from tqdm import tqdm

def split_data(json_file, train_ratio=0.8, random_seed=42):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dialogues = data['dialogues']
    
    # 안나눠져서 일단 넣어놓음
    print(f"Total num of dialogues: {len(dialogues)}")
    if len(dialogues) <= 1:
        raise ValueError("Dataset is empty or contains only one sample")
    
    train_data, val_data = train_test_split(
        dialogues,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    return train_data, val_data

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=128):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length
            
        if not hasattr(DialogueDataset, 'bert'):
            DialogueDataset.bert = AutoModel.from_pretrained('skt/kobert-base-v1')
            DialogueDataset.bert.eval()
        
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        utterances = dialogue['utterances']
        
        texts = [u['text'] for u in utterances]
        speaker_ids = [ord(u['speaker']) - ord('A') for u in utterances]  # A, B, C -> 0, 1, 2
        empathy_scores = [float(u['empathy_score']) for u in utterances]
        
        encoded_texts = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                    # return_token_type_ids=True
                )
                inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
                if 'token_type_ids' not in inputs:
                    inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
                if hasattr(self, 'device'):
                    inputs = {k: v.to(next(self.bert.parameters()).device) for k, v in inputs.items()}
                try:
                    outputs = DialogueDataset.bert(**inputs)
                    # squeeze() 전에 차원 확인
                    hidden_states = outputs.last_hidden_state.mean(dim=1)
                    if hidden_states.dim() > 1:
                        hidden_states = hidden_states.squeeze()
                    encoded_texts.append(hidden_states)
                except Exception as e:
                    print(f"Error processing text: {text}")
                    print(f"Input shape: {inputs['input_ids'].shape}")
                    print(f"Token type IDs shape: {inputs['token_type_ids'].shape}")
                    raise e
                
        # 텐서로 변환
        utterance_vectors = torch.stack(encoded_texts)
        speaker_ids = torch.tensor(speaker_ids)
        empathy_scores = torch.tensor(empathy_scores)
        
        return utterance_vectors, speaker_ids, empathy_scores

def create_data_loaders(train_file, train_ratio=0.8, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    # tokenizer 설정 확인용 코드
    tokenizer.model_max_length=128
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.sep_token is None:
        tokenizer.sep_token == '[SEP]'
        tokenizer.sep_token_id = tokenizer.sep_token_id
    if tokenizer.cls_token is None:
        tokenizer.cls_token = '[CLS]'
        tokenizer.cls_token_id = tokenizer.eos_token_id

    # tokenizer.model_max_length = 1024
    # tokenizer.init_kwargs['model_max_length'] = 1024
    train_data, val_data = split_data(train_file, train_ratio)
    train_dataset = DialogueDataset(train_data, tokenizer)
    val_dataset = DialogueDataset(val_data, tokenizer)
    
    batch_size = min(batch_size, len(train_data))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    return train_loader, val_loader

def collate_fn(batch):
    """배치 데이터 처리를 위한 콜레이트 함수"""
    utterance_vectors, speaker_ids, empathy_scores = zip(*batch)
    
    # 패딩을 위함
    max_len = max(u.size(0) for u in utterance_vectors)
    
    padded_utterances = torch.zeros(len(batch), max_len, utterance_vectors[0].size(-1))
    padded_speaker_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_empathy_scores = torch.zeros(len(batch), max_len)
    
    for i, (u, s, e) in enumerate(zip(utterance_vectors, speaker_ids, empathy_scores)):
        padded_utterances[i, :u.size(0)] = u
        padded_speaker_ids[i, :s.size(0)] = s
        padded_empathy_scores[i, :e.size(0)] = e
    
    return padded_utterances, padded_speaker_ids, padded_empathy_scores

def predict_empathy(model, tokenizer, text, device):
    """단일 텍스트에 대한 공감 점수 예측"""
    model.eval()
    
    # 입력 텍스트 처리
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt',
        return_token_type_ids=True
    )
    if 'token_type_ids' not in inputs:
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
    
    # BERT를 통한 텍스트 인코딩
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = DialogueDataset.bert(**inputs)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
    
    # 모델 예측
    utterance_vector = hidden_states.unsqueeze(0)  # (1, 1, hidden_size)
    speaker_ids = torch.tensor([0]).to(device)  # 단일 화자 가정
    
    with torch.no_grad():
        prediction = model(utterance_vector, speaker_ids)
    
    return prediction.item()

def test_dialogue_example(model, tokenizer, device):
    test_dialogues = [
        ("오늘 발표했는데 긴장해서 실수했어...", "많이 긴장되었겠다. 준비 열심히 했는데 아쉽겠네."),
        ("새로운 취미를 시작해보고 싶은데 뭐가 좋을까?", "요즘 힘들어서 기분 전환이 필요했구나. 같이 찾아볼까?"),
        ("일이 너무 많아서 지쳐요.", "많이 힘들었겠다. 쉬엄쉬엄 하는 게 어떨까?")
    ]
    
    print("\n=== 테스트 대화 예시 ===")
    for speaker1, speaker2 in test_dialogues:
        print("\n화자 A:", speaker1)
        print("화자 B:", speaker2)
        empathy_score = predict_empathy(model, tokenizer, speaker2, device)
        print(f"예측된 공감 점수: {empathy_score:.3f}")
        
        if empathy_score <= 0.3:
            print("해석: 낮은 공감 수준")
        elif empathy_score < 0.8:
            print("해석: 중간 공감 수준")
        else:
            print("해석: 높은 공감 수준")

def main():
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 768  # BERT hidden size
    hidden_size = 256
    num_speakers = 3  
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    
    train_file = "/Users/alice.kim/Desktop/aa/Final/app/services/empathy_dataset.json"

    train_data, val_data = split_data(train_file, train_ratio=0.8)

    train_loader, val_loader = create_data_loaders(
        train_file=train_file,
        batch_size=batch_size,
        train_ratio=0.8  
    )
    num_layers = 1
    model = DialogueEmpathyModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_speakers=num_speakers,
        num_layers=num_layers,
        dropout=0.5 if num_layers > 1 else 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = EmpathyLoss()
    
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    test_dataset = DialogueDataset(val_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_loss, test_mse, test_mae=eval_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

    test_dialogue_example(model, tokenizer, device)

    while True:
        user_input = input("\n테스트할 발화를 입력하세요 (종료하려면 'q' 입력): ")
        if user_input.lower() == 'q':
            break
            
        empathy_score = predict_empathy(model, tokenizer, user_input, device)
        print(f"예측된 공감 점수: {empathy_score:.3f}")
        
        if empathy_score <= 0.3:
            print("해석: 낮은 공감 수준")
        elif empathy_score < 0.8:
            print("해석: 중간 공감 수준")
        else:
            print("해석: 높은 공감 수준")

if __name__ == "__main__":
    main()