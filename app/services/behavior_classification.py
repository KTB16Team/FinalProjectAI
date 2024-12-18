import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn import CrossEntropyLoss, BatchNorm1d
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import re
import requests
import os

class Config:
    MAX_LENGTH = 256
    BATCH_SIZE = 16 #안정성을 위해서 32로 해도 될라나
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    NUM_LABELS = 5
    SAVE_PATH = "Behavior_classifier.pt" 
    PATIENCE = 5
    WEIGHT_DECAY = 0.01

class CustomBERTClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = torch.nn.Dropout(0.1)
        self.bert_config = self.bert.config
        hidden_size = self.bert.config.hidden_size
        # 더 안정적인 분류기 구조
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, Config.NUM_LABELS), labels.view(-1))
        
        return logits, loss

class OverfittingMonitor:
    def __init__(self, patience=3, threshold=0.1):
        self.patience = patience
        self.threshold = threshold
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.overfitting_count = 0
        self.best_val_loss = float('inf')
    
    def check_overfitting(self, train_loss, val_loss, val_f1):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_f1_scores.append(val_f1)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.overfitting_count = 0
            return False
        
        if (train_loss < val_loss) and (val_loss - train_loss > self.threshold):
            self.overfitting_count += 1
        else:
            self.overfitting_count = 0
        return self.overfitting_count >= self.patience
    
    def plot_learning_curves(self):
        plt.figure(figsize=(15,5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_f1_scores, label='Validation F1 Score')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def get_overfitting_metrics(self):
        if len(self.train_losses) > 1:
            avg_train_loss = np.mean(self.train_losses[-3:])  
            avg_val_loss = np.mean(self.val_losses[-3:])
            loss_diff = avg_val_loss - avg_train_loss # 검증-학습 차이 크면 과적합 가능성 높으므로!
            
            return {
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'loss_difference': loss_diff,
                'overfitting_trend': loss_diff > self.threshold,
                'consecutive_overfitting': self.overfitting_count
            }
        return None
    
class BehaviorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_and_prepare_data(file_path, test_size=0.2, random_state=42):
    """데이터 로딩과 분할을 개선한 함수"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts, labels = [], []
    label_map = {category['category']: i for i, category in enumerate(data['behavior_examples'])}
    reverse_label_map = {i: category for category, i in label_map.items()}
    
    for category in data['behavior_examples']:
        for example in category['examples']:
            texts.append(example['text'])
            labels.append(label_map[category['category']])
    print(f"\n원본 데이터 총 개수: {len(texts)}개")


    # 계층적 분할 수행
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, 
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    print("\n검증 데이터 클래스 분포:")
    val_distribution = Counter(val_labels)
    for label_idx in sorted(val_distribution.keys()):
        category_name = reverse_label_map[label_idx]
        print(f"{category_name}: {val_distribution[label_idx]}개")

    return train_texts, val_texts, train_labels, val_labels, label_map

def calculate_class_weights(labels):
    """클래스별 가중치 계산"""
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    print(f"Class weights: {class_weights}")
    return class_weights

def eda_augment(text, alpha=0.1):
    """Easy Data Augmentation (EDA)를 이용한 텍스트 증강"""

    words = text.split()
    # num_words = len(words)

    # 랜덤하게 단어 삭제
    if random.random() < alpha:
        words = [word for word in words if random.random() > 0.1]

    # 랜덤하게 단어 교체
    if random.random() < alpha:
        for i in range(len(words)):
            if random.random() < 0.1:
                synonyms = get_synonyms(words[i])
                if synonyms:  # 유의어가 있는 경우에만 교체
                    words[i] = random.choice(synonyms)

    # 랜덤하게 단어 추가
    if random.random() < alpha:
        for i in range(len(words)):
            if random.random() < 0.1:
                synonyms = get_synonyms(words[i])
                if synonyms:  # 유의어가 있는 경우에만 추가
                    words.insert(i, random.choice(synonyms))

    # 랜덤하게 단어 순서 변경
    if random.random() < alpha:
        random.shuffle(words)

    return ' '.join(words)

def get_synonyms(word):
    """Datamuse API를 사용하여 단어의 유의어 목록 가져오기"""
    url = f"https://api.datamuse.com/words?rel_syn={word}&max=5"
    response = requests.get(url)
    data = response.json()
    synonyms = [d['word'] for d in data]
    
    # 유의어가 없을 경우 원래 단어를 반환하도록 처리
    return synonyms if synonyms else [word]

def similar_embedding_replacement(text, model, tokenizer, alpha=0.1):
    words = text.split()
    new_words = []

    for word in words:
        if random.random() < alpha:
            # 단어 임베딩 가져오기
            word_embedding = model.get_input_embeddings()(tokenizer.encode(word, return_tensors='pt')).squeeze()
            
            # 유사도 계산 및 상위 k개 단어 선택
            similar_words, _ = torch.topk(torch.matmul(model.get_input_embeddings().weight, word_embedding.T), k=5, dim=0)
            
            # 선택된 단어 중 하나를 랜덤하게 대체
            new_word = tokenizer.decode(similar_words[torch.randint(0, 5, (1,)).item()])
            new_words.append(new_word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

def contextualized_embedding_replacement(text, model, tokenizer, alpha=0.1):
    """Contextualized Embedding을 이용한 단어 대체"""
    words = text.split()
    new_words = []

    for i, word in enumerate(words):
        if random.random() < alpha:
            # 현재 단어를 제외한 나머지 문장을 입력으로 사용
            context = words[:i] + words[i+1:]
            context_input = tokenizer(' '.join(context), return_tensors='pt')
            masked_input = context_input.input_ids.clone()
            masked_input[0, i] = tokenizer.mask_token_id
            output = model(**context_input)[0]
            _, indices = torch.topk(output[0, i], k=5)
            
            # 선택된 단어 중 하나를 랜덤하게 대체
            new_word = tokenizer.decode(indices[random.randint(0, 4)])
            new_words.append(new_word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

def augment_data(texts, labels, augmentation_multiplier=2, max_ratio=0.8):
    augmented_texts = []
    augmented_labels = []

    augmented_texts.extend(texts)
    augmented_labels.extend(labels)

    label_names = {
        0: "경쟁형",
        1: "회피형",
        2: "수용형",
        3: "타협형",
        4: "협력형"
    }

    print("\n원본 데이터 클래스 분포:")
    class_distribution = Counter(labels)
    for label_idx in sorted(class_distribution.keys()):
        category_name = label_names[label_idx]
        print(f"{category_name}: {class_distribution[label_idx]}개")

    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    for label in tqdm(class_distribution.keys(), desc="Augmenting data"):
        class_texts = [text for text, lab in zip(texts, labels) if lab == label]
        current_samples = class_distribution[label]

        # EDA 기반 증강
        for text in class_texts:
            for _ in range(int(current_samples * (augmentation_multiplier - 1) / len(class_texts))):
                augmented_text = eda_augment(text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

        # 유사 임베딩 기반 증강
        for text in class_texts:
            for _ in range(int(current_samples * (augmentation_multiplier - 1) / len(class_texts))):
                augmented_text = similar_embedding_replacement(text, bert_model, bert_tokenizer)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

        # Contextualized Embedding 기반 증강
        for text in class_texts:
            for _ in range(int(current_samples * (augmentation_multiplier - 1) / len(class_texts))):
                augmented_text = contextualized_embedding_replacement(text, bert_model, bert_tokenizer)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

    print("\n증강 후 클래스 분포:")
    final_distribution = Counter(augmented_labels)
    for label in sorted(final_distribution.keys()):
        print(f"클래스 {label}: {final_distribution[label]}개")
    
    print(f"\n총 데이터 수 변화: {len(texts)} -> {len(augmented_texts)}")
    print(f"증강된 데이터 수: {len(augmented_texts) - len(texts)}")
    return augmented_texts, augmented_labels

def plot_confusion_matrix(y_true, y_pred, labels):
    """혼동 행렬 시각화 함수"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(model, train_loader, val_loader, device, class_weights, monitor):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

    # best_f1 = 0.0
    # best_model_state = None
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device)
        avg_val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro_f1']
        # 일반화 되기 전에 과적합이 빨리 발생해서 에폭 끝까지 돌려보고자 주석 처리함.
        # is_overfitting = monitor.check_overfitting(avg_train_loss, avg_val_loss, val_f1)
        
        print(f"\n에포크 {epoch + 1}/{Config.EPOCHS}")
        print(f"학습 손실: {avg_train_loss:.4f}")
        print(f"검증 손실: {avg_val_loss:.4f}")
        print(f"검증 F1 점수: {val_f1:.4f}")

    #     if val_f1 > best_f1:
    #         best_f1 = val_f1
    #         best_model_state = model.state_dict().copy()
    #         print(f"새로운 최고 모델 저장 (F1: {val_f1:.4f})")
    #         torch.save(model.state_dict(), Config.SAVE_PATH)

    #     if is_overfitting:
    #         print("\n경고: 과적합 감지! 학습을 조기 종료합니다.")
    #         print("과적합 메트릭:")
    #         metrics = monitor.get_overfitting_metrics()
    #         print(f"- 평균 학습 손실: {metrics['avg_train_loss']:.4f}")
    #         print(f"- 평균 검증 손실: {metrics['avg_val_loss']:.4f}")
    #         print(f"- 손실 차이: {metrics['loss_difference']:.4f}")
    #         print(f"- 연속 과적합 횟수: {metrics['consecutive_overfitting']}")
    #         break
        
        scheduler.step()
        
    monitor.plot_learning_curves()
    
    # final_metrics = monitor.get_overfitting_metrics()
    # if final_metrics:
    #     print("\n=== 최종 과적합 분석 ===")
    #     print(f"최종 평균 학습 손실: {final_metrics['avg_train_loss']:.4f}")
    #     print(f"최종 평균 검증 손실: {final_metrics['avg_val_loss']:.4f}")
    #     print(f"최종 손실 차이: {final_metrics['loss_difference']:.4f}")
    #     print(f"과적합 경향: {'있음' if final_metrics['overfitting_trend'] else '없음'}")
    
    # return best_f1, best_model_state
    return model


def evaluate_model(model, data_loader, device):
    """개선된 모델 평가 함수"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels' : batch['labels'].to(device)
            }
            
            logits, loss = model(**inputs)
            if loss is not None:
                total_loss = loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    # 다양한 메트릭 계산
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # # 혼동 행렬
    # plot_confusion_matrix(all_labels, all_preds, range(Config.NUM_LABELS))
    
    print("\n분류 보고서:")
    print(classification_report(all_labels, all_preds))
    print(f"평슌 검증 손실: {avg_loss:.4f}")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }

def classify_text(model, tokenizer, text, device, label_map):
    """Classify a single input text."""
    model.eval()
    reverse_label_map = {v: k for k, v in label_map.items()}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=Config.MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        logits, _ = model(**inputs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predictions].item()

    return reverse_label_map[predictions], confidence


def map_category_score(category):
    """카테고리별 점수를 매핑"""
    score_map = {
        "경쟁": 0,
        "회피": 0,
        "타협": 0.5,
        "협력": 1,
        "수용": 1
    }
    return score_map.get(category, 0) 

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    data_file = "/Users/alice.kim/Desktop/aa/Final/app/services/behavior_dataset.json"

    train_texts, val_texts, train_labels, val_labels, label_map = load_and_prepare_data(data_file)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print("클래스 가중치:", class_weights)

    train_texts, train_labels = augment_data(train_texts, train_labels, augmentation_multiplier=3)
    print(f"증강 후 총 샘플 수: {len(train_texts)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)

    model.classifier = torch.nn.Sequential(
        BatchNorm1d(model.bert_config.hidden_size),
        model.classifier
    )

    train_dataset = BehaviorDataset(train_texts, train_labels, tokenizer)
    val_dataset = BehaviorDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        drop_last=True
        )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n사용 중인 디바이스: {device}")
    
    monitor = OverfittingMonitor(patience=3, threshold=0.1)
    
    best_f1 = train_model(model, train_loader, val_loader, device, class_weights, monitor)
    print(f"\n학습 완료. 최고 검증 정확도: {best_f1:.4f}")
    
    checkpoint = torch.load(Config.SAVE_PATH, weights_only=True)
    model.load_state_dict(torch.load(Config.SAVE_PATH, weights_only=True))

    print("\n최종 모델 평가:")
    final_metrics = evaluate_model(model, val_loader, device)
    # 모델 테스트
    test_text = "그래 내가 잘못한 거 맞아. 근데 너도 잘못한게 없지는 않잖아."
    prediction, confidence = classify_text(model, tokenizer, test_text, device, label_map)
    category_score = map_category_score(prediction)

    print(f"\n입력 텍스트: {test_text}")
    print(f"예측된 카테고리: {prediction} (확신도: {confidence:.2%})")
    print(f"카테고리 점수: {category_score}")
    
    print("\n모델 저장 경로:", Config.SAVE_PATH)
    
    print("\n=== 학습 결과 요약 ===")
    print(f"총 학습 데이터: {len(train_texts)}개")
    print(f"총 검증 데이터: {len(val_texts)}개")
    print(f"최종 정확도: {final_metrics['accuracy']:.4f}")
    print(f"최종 Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"최종 Weighted F1: {final_metrics['weighted_f1']:.4f}")