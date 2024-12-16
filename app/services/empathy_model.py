# empathy_model.py

import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os

class DialogueEmpathyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_speakers=10, num_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_speakers = num_speakers
        self.num_layers = num_layers
        
        # Text Encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Speaker Embeddings
        self.speaker_embedding = nn.Embedding(num_speakers, hidden_size)
        
        # Utterance Encoder
        self.utterance_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # Context Processor
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Response Analyzer
        self.response_analyzer = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Empathy Classifier
        self.empathy_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Layer Normalizations
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        self.norm3 = nn.LayerNorm(hidden_size * 2)

    def forward(self, utterances, speaker_ids, attention_mask=None):
        batch_size, seq_len, _ = utterances.size()
        device = utterances.device
        
        # 1. Text Encoding with context weight
        encoded_text = self.text_encoder(utterances)
        
        # 2. Speaker Information with role consideration
        speaker_features = self.speaker_embedding(speaker_ids)
        combined_features = encoded_text + speaker_features
        
        # 3. Utterance Encoding with memory
        utterance_features, _ = self.utterance_encoder(combined_features)
        utterance_features = self.norm1(utterance_features)
        
        # 4. Context Attention with history weighting
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        context_features, _ = self.context_attention(
            utterance_features,
            utterance_features,
            utterance_features,
            attn_mask=mask
        )
        context_features = self.norm2(context_features)
        
        # 5. Feature Fusion with enhanced context
        combined = torch.cat([utterance_features, context_features], dim=-1)
        fused_features = self.fusion_layer(combined)
        
        # 6. Response Analysis with deeper understanding
        response_features, _ = self.response_analyzer(fused_features)
        response_features = self.norm3(response_features)
        
        # 7. Final Prediction with confidence weighting
        empathy_scores = self.empathy_classifier(response_features)
        
        return empathy_scores

class EmpathyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions, targets, mask=None):
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)
            
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        
        loss = self.mse_loss(predictions, targets)
        return loss

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=256):
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
        speaker_ids = [ord(u['speaker']) - ord('A') for u in utterances]
        empathy_scores = [float(u['empathy_score']) for u in utterances]
        
        encoded_texts = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
                
                outputs = DialogueDataset.bert(**inputs)
                hidden_states = outputs.last_hidden_state.mean(dim=1)
                
                if hidden_states.dim() > 1:
                    hidden_states = hidden_states.squeeze()
                encoded_texts.append(hidden_states)
                
        utterance_vectors = torch.stack(encoded_texts)
        speaker_ids = torch.tensor(speaker_ids)
        empathy_scores = torch.tensor(empathy_scores)
        
        return utterance_vectors, speaker_ids, empathy_scores

def train_model(model, train_loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0
    epoch_losses = []
    
    for batch_idx, (utterances, speaker_ids, empathy_scores) in enumerate(train_loader):
        utterances = utterances.to(device)
        speaker_ids = speaker_ids.to(device)
        empathy_scores = empathy_scores.to(device)
        
        mask = (empathy_scores != 0)
        
        optimizer.zero_grad()
        predictions = model(utterances, speaker_ids)
        
        loss = criterion(predictions, empathy_scores, mask)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        epoch_losses.append(loss.item())
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss, epoch_losses

def eval_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for utterances, speaker_ids, empathy_scores in test_loader:
            utterances = utterances.to(device)
            speaker_ids = speaker_ids.to(device)
            empathy_scores = empathy_scores.to(device)
            
            predictions = model(utterances, speaker_ids)
            
            if predictions.dim() == 3:
                predictions = predictions.squeeze(-1)

            for pred, target in zip(predictions, empathy_scores):
                valid_length = (target != 0).sum()
                all_predictions.append(pred[:valid_length])
                all_targets.append(target[:valid_length])
            
            loss = criterion(predictions, empathy_scores)
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(empathy_scores.cpu())

    all_predictions = torch.cat([p.view(-1) for p in all_predictions])
    all_targets = torch.cat([t.view(-1) for t in all_predictions])
    
    mse = F.mse_loss(all_predictions, all_targets).item()
    mae = F.l1_loss(all_predictions, all_targets).item()
    
    return total_loss / len(test_loader), mse, mae

def predict_empathy(model, tokenizer, text, device):
    model.eval()

    if not hasattr(DialogueDataset, 'bert'):
        DialogueDataset.bert = AutoModel.from_pretrained('skt/kobert-base-v1')
        DialogueDataset.bert.eval()

    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = DialogueDataset.bert(**inputs)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        
        utterance_vector = hidden_states.unsqueeze(0)
        speaker_ids = torch.tensor([[1]]).to(device)  # B의 응답이므로 speaker_id = 1
        prediction = model(utterance_vector, speaker_ids)
        score = prediction.squeeze().item()
        
        # Add post-processing for more distinct predictions
        score = adjust_score(text, score)
        return round(score, 2)

def adjust_score(text, score):
    """공감 점수 후처리"""
    # 긍정적 공감 표현
    high_empathy = ['이해', '공감', '힘들', '수고', '괜찮', '함께', '같이', '진짜']
    # 부정적/무관심 표현
    low_empathy = ['뭐', '어떡해', '그냥', '알아서']
    
    # 점수 조정
    if any(word in text for word in high_empathy):
        score = min(0.95, score * 1.2)
    if any(word in text for word in low_empathy):
        score = max(0.1, score * 0.8)
        
    return score

def collate_fn(batch):
    utterance_vectors, speaker_ids, empathy_scores = zip(*batch)
    
    max_len = max(u.size(0) for u in utterance_vectors)
    
    padded_utterances = torch.zeros(len(batch), max_len, utterance_vectors[0].size(-1))
    padded_speaker_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_empathy_scores = torch.zeros(len(batch), max_len)
    
    for i, (u, s, e) in enumerate(zip(utterance_vectors, speaker_ids, empathy_scores)):
        padded_utterances[i, :u.size(0)] = u
        padded_speaker_ids[i, :s.size(0)] = s
        padded_empathy_scores[i, :e.size(0)] = e
    
    return padded_utterances, padded_speaker_ids, padded_empathy_scores

def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    MAX_GRAD_NORM = 1.0
    NUM_EPOCHS = 10
    INPUT_SIZE = 768
    HIDDEN_SIZE = 256
    NUM_SPEAKERS = 10
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 더미 데이터 생성 (실제 데이터셋으로 대체 필요)
    def generate_dummy_dialogues(num_dialogues=100):
        dialogues = []
        for _ in range(num_dialogues):
            utterances = [
                {'text': f'대화 테스트 문장 {_}', 'speaker': 'A', 'empathy_score': 0.5},
                {'text': f'응답 테스트 문장 {_}', 'speaker': 'B', 'empathy_score': 0.7}
            ]
            dialogues.append({'utterances': utterances})
        return dialogues
    
    # 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    
    # 더미 데이터셋 생성
    dialogues = generate_dummy_dialogues()
    
    # 데이터셋 분할
    dataset = DialogueDataset(dialogues, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 모델 초기화
    model = DialogueEmpathyModel(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        num_speakers=NUM_SPEAKERS
    ).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = EmpathyLoss()
    optimizer = Adam(model.parameters, lr=LEARNING_RATE)

    os.makedirs('checkpoints', exist_ok=True)
    
    # 학습 루프
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # 학습
        train_loss, _ = train_model(
            model, train_loader, optimizer, criterion, device, MAX_GRAD_NORM
        )
        
        # 검증
        val_loss, val_mse, val_mae = eval_model(
            model, val_loader, criterion, device
        )
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MSE: {val_mse:.4f}")
        print(f"Val MAE: {val_mae:.4f}")
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, 'checkpoints/best_model.pt')
    
    # 테스트 케이스
    test_cases = [
        ("오늘 발표했는데 긴장해서 실수했어...", "헐 발표면 진짜 많이 긴장되었겠다. 실수는 누구나 할 수 있지! 담에 잘하면 되지. 너무 수고했다 진짜!"),
        ("새로운 취미를 시작해보고 싶은데 뭐가 좋을까?", "요즘 힘들어서 기분 전환이 필요했구나. 같이 찾아볼까?"),
        ("일이 너무 많아서 지쳐요.", "뭐 어떡해 해야지.")
    ]
    
    # 모델 로드 및 테스트
    model = load_model('checkpoints/best_model.pt', device)
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    
    print("\n=== 모델 예측 테스트 ===")
    for speaker1, speaker2 in test_cases:
        print(f"\nA: {speaker1}")
        print(f"B: {speaker2}")
        
        score = predict_empathy(model, tokenizer, speaker2, device)
        print(f"예측된 공감 점수: {score:.2f}")
        
        if score <= 0.39:
            print("낮은 공감 수준")
        elif score <= 0.79:
            print("중간 공감 수준")
        else:
            print("높은 공감 수준")

def load_model(model_path, device='cpu'):
    """모델 로드 함수"""
    model = DialogueEmpathyModel(
        input_size=768,
        hidden_size=256,
        num_speakers=10,
        num_layers=1,
        dropout=0.5
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

if __name__ == "__main__":
    main()