import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class DialogueEmpathyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_speakers, num_layers=1, dropout=0.5):
        super(DialogueEmpathyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_speakers = num_speakers
        
        self.initial_party_state = nn.Parameter(torch.zeros(1,1,hidden_size))
        self.initial_global_state = nn.Parameter(torch.zeros(1, num_speakers, hidden_size))
        self.initial_emotion_state = nn.Parameter(torch.zeros(1,1,hidden_size))

        # Party GRU (맥락 파악용)
        # num_layers = 1 #원하는 layer 수 
        self.party_gru = nn.GRU(
            input_size + hidden_size,  # 발언 벡터 + 맥락 벡터
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Global GRU (화자 상태 추적용)
        self.global_gru = nn.GRU(
            input_size + hidden_size,  # 발언 벡터 + 화자 현재 상태
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Emotion GRU (공감 지수 예측용)
        self.emotion_gru = nn.GRU(
            2 * hidden_size,  # 이전 emotion GRU 결과 + 화자 현재 상태
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0

        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size//2),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # 초기화
        # nn.init.uniform_(self.context_vector, -0.1, 0.1)
        
    def attention_net(self, party_hidden, mask=None):
        attention_weights = self.attention(party_hidden)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~mask.unsqueeze(-1),
                float('-inf')
            )
            
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), party_hidden)
        return context.squeeze(1)
    
    def forward(self, utterances, speaker_ids, attention_mask = None):
        
        batch_size, seq_len, _ = utterances.size()
        device = utterances.device
        
        party_hidden = self.initial_party_state.expand(batch_size, -1, -1).to(device)
        global_hidden =self.initial_global_state.expand(batch_size, -1, -1).to(device)
        emotion_hidden = self.initial_emotion_state.expand(batch_size, -1, -1).to(device)
        
        empathy_scores = []
        
        for t in range(seq_len):
            if attention_mask is not None and not attention_mask[:, t].any():
                continue
            curr_utterance = utterances[:, t:t+1, :]
            curr_speaker = speaker_ids[:, t]
            
            # 1.
            context = self.attention_net(
                party_hidden,
                attention_mask[:, :t+1] if attention_mask is not None else None
            )
            party_input = torch.cat([curr_utterance, context.unsqueeze(1)], dim=-1)
            party_output, _ = self.party_gru(party_input, party_hidden.transpose(0,1))
            party_hidden = self.layer_norm(party_output)
            
            # 2.
            new_global_hidden = global_hidden.clone()
            speaker_states = []
            for s in range(self.num_speakers):
                is_current_speaker = (curr_speaker == s).float().unsqueeze(1).unsqueeze(2)
                global_input = torch.cat([curr_utterance, global_hidden[:, s:s+1, :]], dim=-1)
                global_output, _ = self.global_gru(global_input, global_hidden[:, s:s+1, :].transpose(0,1))
                normalized_output = self.layer_norm(global_output)

                new_global_hidden[:, s:s+1, :] = normalized_output * is_current_speaker + \
                global_hidden[:, s:s+1, :] * (1-is_current_speaker)
            global_hidden = new_global_hidden 


            speaker_states = [global_hidden[:, s:s+1, :] for s in range(self.num_speakers)]
            curr_speaker_state = torch.stack([speaker_states[s.item()] for s in curr_speaker], dim=1)
            # 현재 화자의 상태 가져오기
            curr_speaker_state = torch.stack([speaker_states[s.item()] for s in curr_speaker], dim=1)
            
            # 3. Emotion GRU: 공감 지수 예측
            emotion_input = torch.cat([emotion_hidden, curr_speaker_state], dim=-1)
            emotion_output, _ = self.emotion_gru(emotion_input, emotion_hidden.transpose(0,1))
            emotion_hidden = self.layer_norm(emotion_output)
            
            empathy_score = self.output_layer(emotion_output.squeeze(1))
            empathy_scores.append(empathy_score)
        
        return torch.stack(empathy_scores, dim=1)

class EmpathyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, predictions, targets, mask=None):
        if mask is not None:
            masked_pred = predictions[mask]
            masked_targets = targets[mask]
            return self.mse_loss(masked_pred, masked_targets)
        return self.mse_loss(predictions, targets)

def train_model(model, train_loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0
    
    for batch_idx, (utterances, speaker_ids, empathy_scores) in enumerate(train_loader):
        utterances = utterances.to(device)
        speaker_ids = speaker_ids.to(device)
        empathy_scores = empathy_scores.to(device)
        
        optimizer.zero_grad()
        predictions = model(utterances, speaker_ids)
        loss = criterion(predictions, empathy_scores)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

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
            
            loss = criterion(predictions, empathy_scores)
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(empathy_scores.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    mse = F.mse_loss(all_predictions, all_targets).item()
    mae = F.l1_loss(all_predictions, all_targets).item()

    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, mse, mae