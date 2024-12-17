import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel,pipeline
import numpy as np
import json
import random
from typing import List, Dict, Tuple
# from googletrans import Translator
# from konlpy.tag import Okt
# import nltk
# from nltk.corpus import wordnet

class TextAugmenter:
    def __init__(self):
        # self.translator = Translator()
        
    # def back_translation(self, text: str, middle_lang='en') -> str:
    #     try:
    #         mid_text = self.translator.translate(text, dest=middle_lang).text
    #         return self.translator.translate(mid_text, dest='ko').text
    #     except:
    #         return text
        pass
    def random_insertion(self, text: str, n_words: int = 1) -> str:
        words = text.split()
        for _ in range(n_words):
            insert_pos = random.randint(0, len(words))
            insert_word = random.choice(['갑자기', '정말', '매우', '아마도', '확실히'])
            words.insert(insert_pos, insert_word)
        return ' '.join(words)

    def add_noise(self, text: str, p: float = 0.1) -> str:
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p:
                noise_type = random.choice(['swap', 'delete', 'insert'])
                if noise_type == 'swap' and i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                elif noise_type == 'delete':
                    chars[i] = ''
                elif noise_type == 'insert':
                    chars.insert(i, random.choice('가나다라마바사아자차카타파하'))
        return ''.join(chars)

    def augment_conversation(self, conversation: Dict) -> List[Dict]:
        augmented_conversations = []
        augmented_conversations.append(conversation)  # 원본 보존
        
        augmentation_methods = [
            self.back_translation,
            self.random_insertion,
            self.add_noise,
        ]
        
        for method in augmentation_methods:
            new_conv = conversation.copy()
            new_conv = {k: v.copy() if isinstance(v, dict) else v for k, v in conversation.items()}
            new_utterances = []
            
            for utterance in conversation['utterances']:
                try:
                    new_utterance = method(utterance)
                    new_utterances.append(new_utterance)
                except:
                    new_utterances.append(utterance)
            
            new_conv['utterances'] = new_utterances
            new_conv['conversation_id'] = f"{conversation['conversation_id']}_{method.__name__}"
            augmented_conversations.append(new_conv)
            
        return augmented_conversations

class AugmentedConversationDataset(Dataset):
    def __init__(self, conversations: List[Dict], tokenizer: AutoTokenizer, augmenter: TextAugmenter = None):
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        
        if augmenter:
            augmented_conversations = []
            for conv in conversations:
                augmented_conversations.extend(augmenter.augment_conversation(conv))
            self.conversations = augmented_conversations
        else:
            self.conversations = conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        conversation = self.conversations[idx]
        utterances = conversation['utterances']
        max_utterances = max(len(conv['utterances']) for conv in self.conversations)
        
        padded_utterances = utterances + ['[PAD]'] * (max_utterances - len(utterances))
        padded_emotions = conversation['emotions'] + [0.0] * (max_utterances - len(utterances))
        
        encodings = self.tokenizer(
            padded_utterances,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        attention_mask = encodings['attention_mask']
        utterance_mask = torch.ones(max_utterances)
        utterance_mask[len(utterances):] = 0

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': attention_mask,
            'emotion_labels': torch.tensor(padded_emotions, dtype=torch.float),
            'utterance_mask': utterance_mask,
            'conversation_id': conversation['conversation_id'],
            'situation': conversation['context_labels']['situation'],
            'emotion_flow': conversation['context_labels']['emotion_flow']
        }

def prepare_augmented_datasets(dataset: Dict, train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    conversations = dataset['conversations']
    n_samples = len(conversations)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]
    
    train_conversations = [conversations[i] for i in train_indices]
    valid_conversations = [conversations[i] for i in valid_indices]
    
    return train_conversations, valid_conversations


class ConversationDataset(Dataset):
    def __init__(self, conversations: List[Dict], tokenizer: AutoTokenizer, max_length: int = 64):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        conversation = self.conversations[idx]
        utterances = conversation['utterances']
        max_utterances = max(len(conv['utterances']) for conv in self.conversations)
        
        padded_utterances = utterances + ['[PAD]'] * (max_utterances - len(utterances))
        padded_emotions = conversation['emotions'] + [0.0] * (max_utterances - len(utterances))

        encodings = self.tokenizer(
            padded_utterances,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        attention_mask = encodings['attention_mask']
        utterance_mask = torch.ones(max_utterances)
        utterance_mask[len(utterances):] = 0

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'emotion_labels': torch.tensor(padded_emotions, dtype=torch.float),
            'utterance_mask': utterance_mask,
            'conversation_id': conversation['conversation_id'],
            'situation': conversation['context_labels']['situation'],
            'emotion_flow': conversation['context_labels']['emotion_flow']
        }

class ContextualEmotionAnalyzer(nn.Module):
    def __init__(self, pretrained_model_name: str = 'bert-base-multilingual-cased'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        self.hidden_size = self.bert.config.hidden_size
        
        self.context_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.attention = nn.Linear(self.hidden_size, 1)
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, utterance_mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
    # 입력 차원 처리
        if len(input_ids.shape) == 2:  # [batch_size, seq_length]
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if utterance_mask is not None:
                utterance_mask = utterance_mask.unsqueeze(0)
            batch_size, seq_length, max_length = input_ids.size()
        else:  # [batch_size, seq_length, max_length]
            batch_size, seq_length, max_length = input_ids.size()

        # 텐서를 디바이스로 이동
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if utterance_mask is not None:
            utterance_mask = utterance_mask.to(self.device)

        # BERT 처리를 위해 텐서 평탄화
        flattened_input_ids = input_ids.view(-1, max_length)
        flattened_attention_mask = attention_mask.view(-1, max_length)
        
        # BERT로 문장 임베딩 추출
        bert_outputs = self.bert(
            input_ids=flattened_input_ids,
            attention_mask=flattened_attention_mask
        )
        
        # [CLS] 토큰의 표현을 사용
        sequence_output = bert_outputs.last_hidden_state[:, 0, :]
        sequence_output = sequence_output.view(batch_size, seq_length, -1)
        
        try:
            lstm_output, _ = self.context_lstm(sequence_output)
        except RuntimeError as e:
            print(f"LSTM error: {e}")
            lstm_output = sequence_output
        
        # Attention 가중치 계산
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        if utterance_mask is not None:
            mask = utterance_mask.unsqueeze(-1).bool()
            attention_weights = attention_weights.masked_fill(~mask, 0.0)
            # Renormalize attention weights
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        
    # Ensure correct dimensions for batch matrix multiplication
        attention_weights = attention_weights.view(batch_size, seq_length, 1)
        lstm_output = lstm_output.view(batch_size, seq_length, self.hidden_size)
    
        
        # 문맥을 고려한 출력 생성
        attended_output = torch.bmm(attention_weights.transpose(1, 2), lstm_output)
        attended_output = attended_output.repeat(1, seq_length, 1)
        
        # Combine features
        combined_output = torch.cat([sequence_output, attended_output], dim=2)
        emotion_scores = self.emotion_classifier(combined_output)
        
        if utterance_mask is not None:
            emotion_scores = emotion_scores.masked_fill(~mask, 0.0)
        
        return emotion_scores, attention_weights.squeeze(-1)


class EmotionAnalyzer:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cpu')
        self.model = ContextualEmotionAnalyzer()
        self.tokenizer = self.model.tokenizer
        if model_path:
            # CPU에 모델을 로드하도록 map_location 설정
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        

    def _collate_fn(self, batch):
        max_utterances = max(x['input_ids'].size(0) for x in batch)
        
        batched = {
            'input_ids': [],
            'attention_mask': [],
            'emotion_labels': [],
            'utterance_mask': []
        }
        
        for item in batch:
            curr_len = item['input_ids'].size(0)
            padding_len = max_utterances - curr_len
            
            batched['input_ids'].append(
                F.pad(item['input_ids'], (0, 0, 0, padding_len))
            )
            batched['attention_mask'].append(
                F.pad(item['attention_mask'], (0, 0, 0, padding_len))
            )
            batched['emotion_labels'].append(
                F.pad(item['emotion_labels'], (0, padding_len))
            )
            batched['utterance_mask'].append(
                F.pad(item['utterance_mask'], (0, padding_len))
            )
        
        return {
            'input_ids': torch.stack(batched['input_ids']),
            'attention_mask': torch.stack(batched['attention_mask']),
            'emotion_labels': torch.stack(batched['emotion_labels']),
            'utterance_mask': torch.stack(batched['utterance_mask'])
        }

    def save_model(self, save_path: str):
        """Save the trained model to the specified path."""
        torch.save(self.model.state_dict(), save_path)
        print(f"모델이 {save_path}에 저장되었습니다.")

    def train(self, train_dataset: ConversationDataset, valid_dataset: ConversationDataset = None,
              batch_size: int = 16, epochs: int = 10, learning_rate: float = 2e-5, save_path: str = 'emotion_model.pt'):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=batch_size,
                collate_fn=self._collate_fn
            )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss(reduction='none')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)
                labels = batch['emotion_labels'].to(self.device)
                
                outputs, _ = self.model(input_ids, attention_mask, utterance_mask)
                
                loss = criterion(outputs.squeeze(), labels)
                loss = (loss * utterance_mask).sum() / utterance_mask.sum()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
            
            if valid_dataset:
                valid_loss = self.validate(valid_loader, criterion)
                print(f'Validation Loss: {valid_loss:.4f}')
        
        # Save the model after training
        self.save_model(save_path)

    def validate(self, valid_loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)
                labels = batch['emotion_labels'].to(self.device)
                
                outputs, _ = self.model(input_ids, attention_mask, utterance_mask)
                loss = criterion(outputs.squeeze(), labels)
                loss = (loss * utterance_mask).sum() / utterance_mask.sum()
                
                total_loss += loss.item()
        
        return total_loss / len(valid_loader)

    def analyze_conversation(self, conversation: List[str]) -> List[Dict]:
        self.model.eval()
        
        # Prepare input tensors
        encodings = self.tokenizer(
            conversation,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        ).to(self.device)
        
        # Create utterance mask
        utterance_mask = torch.ones(1, len(conversation)).to(self.device)
        
        with torch.no_grad():
            emotion_scores, attention_weights = self.model(
                encodings['input_ids'],
                encodings['attention_mask'],
                utterance_mask
            )
        
        # Squeeze the outputs properly
        emotion_scores = emotion_scores.squeeze()
        attention_weights = attention_weights.squeeze()
        
        # Handle the case when there's only one utterance
        if len(conversation) == 1:
            emotion_scores = emotion_scores.unsqueeze(0)
            attention_weights = attention_weights.unsqueeze(0)
        
        results = []
        for i, (text, score) in enumerate(zip(conversation, emotion_scores)):
            emotion_type = self._interpret_score(score.item())
            results.append({
                'text': text,
                'emotion_score': score.item(),
                'emotion_type': emotion_type,
                'context_weight': attention_weights[i].item()  # Removed the extra dimension
            })
        
        return results

    def _interpret_score(self, score: float) -> str:
        if score >= 0.8: return "매우 긍정"
        elif score >= 0.6: return "긍정"
        elif score >= 0.4: return "중립"
        elif score >= 0.2: return "부정"
        else: return "매우 부정"

def load_dataset(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"데이터셋 로드 완료: {len(dataset['conversations'])}개의 대화")
        return dataset
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {file_path}")
        return None


def main():
    from transformers.file_utils import is_offline_mode

    is_offline_mode.value = True

    system = EmotionAnalyzer()
    dataset = load_dataset('/content/sample_data/BERT-based_dataset.json')
    
    if dataset is None:
        return
    augmentor = TextAugmenter() 
    train_conversations, valid_conversations = prepare_augmented_datasets(dataset)
    print(f"\n증강 전 데이터 개수:")
    print(f"학습 데이터: {len(train_conversations)}개")
    print(f"검증 데이터: {len(valid_conversations)}개")
    train_dataset = AugmentedConversationDataset(conversations=train_conversations, tokenizer=system.tokenizer, augmenter=augmentor)
    valid_dataset = AugmentedConversationDataset(conversations=valid_conversations, tokenizer=system.tokenizer, augmenter=augmentor)
    print(f"\n증강 후 데이터 개수:")
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(valid_dataset)}개")
    
    print(f"\n증강 배율: {len(train_dataset)/len(train_conversations):.1f}배")
    system.train(train_dataset, valid_dataset)

    test_conversation = [
        "오늘 중요한 시험 결과가 나왔어.",
        "열심히 준비한 만큼 좋은 점수를 받았어!",
        "이제 한시름 놓인다.",
        "오늘 정말 최악의 하루였어. 일이 잘 안 풀리고, 계속 실수를 했어. 너무 힘들고 지쳐서 아무것도 하고 싶지 않아."
    ]
    
    results = system.analyze_conversation(test_conversation)
    
    for result in results:
        print(f"\n텍스트: {result['text']}")
        print(f"감정 점수: {result['emotion_score']:.3f}")
        print(f"감정 유형: {result['emotion_type']}")
        print(f"문맥 가중치: {result['context_weight']:.3f}")

if __name__ == "__main__":
    main()