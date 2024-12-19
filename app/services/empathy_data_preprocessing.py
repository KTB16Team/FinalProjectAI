import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer
from empathy_score import DialogueEmpathyModel, EmpathyLoss,train_model, eval_model
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoModel
from tqdm import tqdm
import random
import requests
import matplotlib.pyplot as plt
import numpy as np
import re

class TextAugmentation:
    def __init__(self):
        # 한국어 유사어 사전 확장
        self.synonym_dict = {
            '좋다': ['훌륭하다', '괜찮다', '멋지다', '만족스럽다'],
            '나쁘다': ['싫다', '안좋다', '형편없다', '불만족스럽다'],
            '크다': ['거대하다', '큼직하다', '넓다', '방대하다'],
            '작다': ['조그맣다', '아담하다', '미세하다', '협소하다'],
            '슬프다': ['우울하다', '속상하다', '마음아프다', '괴롭다'],
            '화나다': ['짜증나다', '열받다', '불쾌하다', '격분하다'],
            '기쁘다': ['행복하다', '즐겁다', '신나다', '황홀하다'],
            '힘들다': ['어렵다', '고되다', '버겁다', '지치다'],
            '걱정': ['불안', '근심', '고민', '염려'],
            '무섭다': ['두렵다', '겁나다', '소름끼치다', '공포스럽다'],
            '공감': ['이해', '동감', '인정', '수긍'],
            '응원': ['격려', '지지', '성원', '지원'],
            '위로': ['안심', '위안', '격려', '안도'],
            '사랑': ['애정', '사랑스러움', '그리움', '그립다'],
            '행복': ['즐거움', '기쁨', '만족', '희열']
        }
        
    def random_deletion(self, text, p=0.1):
        """임의의 단어를 삭제"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        if len(new_words) == 0:
            return random.choice(words)
            
        return ' '.join(new_words)
    
    def random_swap(self, text, n=1):
        """임의의 두 단어의 위치를 교환"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def random_insertion(self, text, n=1):
        """문장 내 임의의 위치에 유사 단어 삽입"""
        words = text.split()
        if not words:
            return text
            
        for _ in range(n):
            insert_pos = random.randint(0, len(words))
            insert_word = random.choice(words)
            words.insert(insert_pos, insert_word)
            
        return ' '.join(words)
    
    def synonym_replacement(self, text, n=1):
        """단순 사전 기반 유사어 치환"""
        words = text.split()
        new_words = []
        
        for word in words:
            # 기본형으로 변환을 시도 (기본형이 사전에 있는 경우)
            for base_word in self.synonym_dict:
                if base_word in word:  # 단어에 기본형이 포함되어 있다면
                    if random.random() < 0.3:  # 30% 확률로 교체
                        word = random.choice(self.synonym_dict[base_word])
                    break
            new_words.append(word)
            
        return ' '.join(new_words)
    
    def noise_injection(self, text, p=0.1):
        """텍스트에 노이즈 주입"""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p:
                # 한글 자음/모음 변형 확장
                noise_dict = {
                    'ㄱ': 'ㅋ', 'ㄷ': 'ㅌ', 'ㅂ': 'ㅍ',
                    'ㅅ': 'ㅎ', 'ㅈ': 'ㅊ', 'ㅇ': 'ㅎ',
                    '요': '용', '죠': '죵', '애': '에',
                    '아': '어', '워': '와', '은': '는',
                    '을': '를', '이': '가'
                }
                if chars[i] in noise_dict:
                    chars[i] = noise_dict[chars[i]]
                    
        return ''.join(chars)
    
    def augment_text(self, text, num_aug=4):
        """여러 증강 기법을 조합하여 적용"""
        augmented_texts = []
        
        for _ in range(num_aug):
            aug_text = text
            
            # 각 증강 기법을 랜덤하게 적용
            if random.random() < 0.3:
                aug_text = self.random_deletion(aug_text)
            if random.random() < 0.3:
                aug_text = self.random_swap(aug_text)
            if random.random() < 0.3:
                aug_text = self.random_insertion(aug_text)
            if random.random() < 0.3:
                aug_text = self.synonym_replacement(aug_text)
            if random.random() < 0.2:
                aug_text = self.noise_injection(aug_text)
                
            if aug_text != text:  # 원본과 다른 경우만 추가
                augmented_texts.append(aug_text)
                
        return augmented_texts

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
        speaker_ids = torch.tensor(speaker_ids).unsqueeze(0)
        speaker_ids = speaker_ids.expand(utterance_vectors.size(0),)
        empathy_scores = torch.tensor(empathy_scores)
        
        return utterance_vectors, speaker_ids, empathy_scores

def create_data_loaders(train_file, train_ratio=0.8, batch_size=16):
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
        # return_token_type_ids=True
    )
    # if 'token_type_ids' not in inputs:
    inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
    
    # BERT를 통한 텍스트 인코딩
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            outputs = DialogueDataset.bert(**inputs)
            hidden_states = outputs.last_hidden_state.mean(dim=1)
        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Input shapes:")
            for k, v in inputs.items():
                print(f"{k}: {v.shape}")
            raise e
    
    # 모델 예측
    utterance_vector = hidden_states.unsqueeze(0)  # (1, 1, hidden_size)
    speaker_ids = torch.tensor([[0]]).to(device)  # 단일 화자 가정
    
    with torch.no_grad():
        prediction = model(utterance_vector, speaker_ids)
        if prediction.dim() == 3:
            prediction = prediction.squeeze(-1)
        prediction = prediction.squeeze()
    
    return prediction.item()

def test_dialogue_example(model, tokenizer, device):
    test_dialogues = [
        ("오늘 발표했는데 긴장해서 실수했어...", "헐 발표면 진짜 많이 긴장되었겠다. 실수는 누구나 할 수 있지! 담에 잘하면 되지. 너무 수고했다 진짜!"),
        ("새로운 취미를 시작해보고 싶은데 뭐가 좋을까?", "요즘 힘들어서 기분 전환이 필요했구나. 같이 찾아볼까?"),
        ("일이 너무 많아서 지쳐요.", "뭐 어떡해 해야지.")
    ]
    
    print("\n=== 테스트 대화 예시 ===")
    for speaker1, speaker2 in test_dialogues:
        print("\nA:", speaker1)
        print("B:", speaker2)
        empathy_score = predict_empathy(model, tokenizer, speaker2, device)
        print(f"예측된 공감 점수: {empathy_score:.2f}")
        
        if empathy_score <= 0.39:
            print("낮은 공감 수준")
        elif empathy_score <= 0.79:
            print("중간 공감 수준")
        else:
            print("높은 공감 수준")

def eda_augment(text, alpha=0.1):
    words = text.split()

    if random.random() < alpha:
        words = [word for word in words if random.random() > 0.1]

    # if random.random() < alpha:
    #     for i in range(len(words)):
    #         if random.random() < 0.1:
    #             synonyms = get_synonyms(words[i])
    #             if synonyms:
    #                 words[i] = random.choice(synonyms)

    # if random.random() < alpha:
    #     for i in range(len(words)):
    #         if random.random() < 0.1:
    #             synonyms = get_synonyms(words[i])
    #             if synonyms:
    #                 words.insert(i, random.choice(synonyms))

    # if random.random() < alpha:
    #     random.shuffle(words)

    return ' '.join(words)

def get_synonyms(word):
    url = f"https://api.datamuse.com/words?rel_syn={word}&max=5"
    response = requests.get(url)
    data = response.json()
    synonyms = [d['word'] for d in data]

    return synonyms if synonyms else [word]

def similar_embedding_replacement(text, model, tokenizer, alpha=0.1):
    words = text.split()
    new_words = []

    for word in words:
        if random.random() < alpha:
            word_embedding = model.get_input_embeddings()(tokenizer.encode(word, return_tensors='pt')).squeeze()
            similar_words, _ = torch.topk(torch.matmul(model.get_input_embeddings().weight, word_embedding.T), k=5, dim=0)
            new_word = tokenizer.decode(similar_words[torch.randint(0, 5, (1,)).item()])
            new_words.append(new_word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

def contextualized_embedding_replacement(text, model, tokenizer, alpha=0.1):
    words = text.split()
    new_words = []

    for i, word in enumerate(words):
        if random.random() < alpha:
            context = words[:i] + words[i+1:]
            context_input = tokenizer(' '.join(context), return_tensors='pt')
            masked_input = context_input.input_ids.clone()
            masked_input[0, i] = tokenizer.mask_token_id
            output = model(**context_input)[0]
            _, indices = torch.topk(output[0, i], k=5)

            new_word = tokenizer.decode(indices[random.randint(0, 4)])
            new_words.append(new_word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

def augment_data(texts, augmentation_multiplier=2):
    augmented_texts = []
    augmenter = TextAugmentation()
    augmented_texts.extend(texts)

    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    for text in tqdm(texts, desc="Augmenting data"):
        for _ in range(augmentation_multiplier - 1):
            augmented_text = eda_augment(text)
            augmented_texts.append(augmented_text)
        
        aug_texts = augmenter.augment_text(text, num_aug = augmentation_multiplier)
        augmented_texts.extend(aug_texts)

        for _ in range(augmentation_multiplier - 1):
            if random.random() < 0.5:
                aug_text = similar_embedding_replacement(text, bert_model, bert_tokenizer)
            else:
                aug_text = contextualized_embedding_replacement(text, bert_model, bert_tokenizer)
            augmented_texts.append(aug_text)


    augmented_texts = list(set(augmented_texts))
    print(f"원본 데이터 크기: {len(texts)}")
    print(f"증강 후 데이터 크기: {len(augmented_texts)}")
    return augmented_texts

def process_augmented_data(train_data, augmented_texts, augmentation_multiplier=2):
    train_data_augmented = []
    # 원본 데이터 먼저 추가
    train_data_augmented.extend(train_data)
    
    text_index = len([u for d in train_data for u in d['utterances']])  # 원본 텍스트 건너뛰기
    
    # 증강된 데이터 추가
    for dialogue in train_data:
        for _ in range(augmentation_multiplier - 1):  # 원본 제외 추가 생성
            augmented_dialogue = {'utterances': []}
            for utterance in dialogue['utterances']:
                augmented_utterance = utterance.copy()
                augmented_utterance['text'] = augmented_texts[text_index]
                text_index += 1
                augmented_dialogue['utterances'].append(augmented_utterance)
            train_data_augmented.append(augmented_dialogue)
    
    return train_data_augmented


def main():
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_size = 768  # BERT hidden size
    hidden_size = 256
    num_speakers = 10
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    
    train_file = "/Users/alice.kim/Desktop/aa/Final/app/services/empathy_dataset.json"

    # 데이터 분할 및 증강
    train_data, val_data = split_data(train_file, train_ratio=0.8)
    
    # 데이터 크기 확인
    print(f"\nInitial data split:")
    print(f"Train data size: {len(train_data)}")
    print(f"Val data size: {len(val_data)}")

    # 데이터 증강
    train_texts = [u['text'] for d in train_data for u in d['utterances']]
    augmentation_multiplier = 4
    
    print("\nStarting data augmentation...")
    augmented_texts = augment_data(train_texts, augmentation_multiplier=augmentation_multiplier)
    train_data_augmented = process_augmented_data(train_data, augmented_texts, augmentation_multiplier)

    print(f"\nAfter augmentation:")
    print(f"Original train data size: {len(train_data)}")
    print(f"Augmented train data size: {len(train_data_augmented)}")

    # DataLoader 생성
    print("\nCreating data loaders...")
    train_dataset_augmented = DialogueDataset(train_data_augmented, tokenizer)
    val_dataset = DialogueDataset(val_data, tokenizer)

    train_loader_augmented = DataLoader(
        train_dataset_augmented,
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

    print(f"Train loader batches: {len(train_loader_augmented)}")
    print(f"Val loader batches: {len(val_loader)}")

    print("\nInitializing model...")
    num_layers = 1
    model = DialogueEmpathyModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_speakers=num_speakers,
        num_layers=num_layers,
        dropout=0.5 if num_layers > 1 else 0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = EmpathyLoss()

    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        avg_train_loss, epoch_losses = train_model(
            model, train_loader_augmented, optimizer, criterion, device, max_grad_norm=1.0
        )
        train_losses.append(avg_train_loss)
        
        # 검증
        val_loss, val_mse, val_mae = eval_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}')
        
        # 최적의 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")

    # 테스트 예시 실행
    print("\nRunning test examples...")
    test_dialogue_example(model, tokenizer, device)

    # 학습 곡선 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()