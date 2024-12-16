import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LinearProjection(nn.Module):
    """차원 변환을 일관되게 처리하기 위한 클래스"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        
    def forward(self, x):
        # 입력이 3차원이 아니면 3차원으로 변환
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 선형 변환 및 정규화
        x = self.linear(x)
        x = self.norm(x)
        return x
    
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
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Layer Normalizations
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        self.norm3 = nn.LayerNorm(hidden_size * 2)


    def forward(self, utterances, speaker_ids, attention_mask=None):
        batch_size, seq_len, _ = utterances.size()
        device = utterances.device
        
        # 1. Text Encoding
        encoded_text = self.text_encoder(utterances)
        
        # 2. Speaker Information
        speaker_features = self.speaker_embedding(speaker_ids)
        combined_features = encoded_text + speaker_features
        
        # 3. Utterance Encoding
        utterance_features, _ = self.utterance_encoder(combined_features)
        utterance_features = self.norm1(utterance_features)
        
        # 4. Context Attention
        # Create padding mask (1 for valid positions, 0 for padding)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        context_features, _ = self.context_attention(
            utterance_features,
            utterance_features,
            utterance_features,
            key_padding_mask=~padding_mask
        )
        context_features = self.norm2(context_features)

        # 5. Feature Fusion
        combined = torch.cat([utterance_features, context_features], dim=-1)
        fused_features = self.fusion_layer(combined)

        # 6. Response Analysis
        response_features, _ = self.response_analyzer(fused_features)
        response_features = self.norm3(response_features)

        # 7. Final Prediction
        empathy_scores = self.empathy_classifier(response_features)

        return empathy_scores

    def _generate_attention_mask(self, seq_len):
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _adjust_scores(self, scores, utterances):
        """Adjust empathy scores based on linguistic features"""
        batch_size, seq_len, _ = utterances.size()
        
        # Convert scores for adjustment
        adjusted = scores.squeeze(-1)
        
        # Normalize scores to ensure reasonable bounds
        min_val = 0.1
        max_val = 0.9
        adjusted = (max_val - min_val) * adjusted + min_val
        
        # Add small noise to break symmetry
        noise = torch.randn_like(adjusted) * 0.02
        adjusted = adjusted + noise
        
        # Ensure scores stay in bounds
        adjusted = torch.clamp(adjusted, min_val, max_val)
        
        return adjusted.unsqueeze(-1)

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
        
        return self.mse_loss(predictions, targets)
 
def train_model(model, train_loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0
    epoch_losses = []
    
    for utterances, speaker_ids, empathy_scores in train_loader:
        utterances = utterances.to(device)
        speaker_ids = speaker_ids.to(device)
        empathy_scores = empathy_scores.to(device)
        
        # Mask for padding
        mask = (empathy_scores != 0)

        optimizer.zero_grad()
        predictions = model(utterances, speaker_ids)
        
        # Apply mask to loss calculation
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

    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, mse, mae