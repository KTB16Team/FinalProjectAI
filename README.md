# FinalProjectAI

AI 기반 중재 서비스를 제공하기 위해 사용자 행동 분류, 감정 분석, 공감 점수 측정 등 세 가지 주요 모델을 통합하여 갈등 상황을 분석하고 과실 비율을 산출하는 프로젝트입니다. 
본 문서는 AI 모델 구조, 데이터 처리 방식, 학습 과정, 그리고 모델 통합 로직을 다룹니다.

---
# 🧶 홍연(紅緣) - Aimo: AI 중재자

**“인연을 이어주는 붉은 실”**  
홍연은 AI 기술을 통해 사람들 간 갈등을 분석하고, 해결 방안을 제시하는 혁신적인 중재 서비스를 제공합니다.

---

## 📖 프로젝트 개요

Aimo는 음성 및 텍스트 데이터를 AI 모델로 분석하여 갈등의 원인을 파악하고 공정한 해결 방안을 제시합니다.  
서비스는 **STT**, **OCR**, **갈등 분석 AI 모델**로 구성되며, 최종적으로 각 참여자의 과실 비율을 산출합니다.

---

## 🧠 AI 모델 로직 및 구현

### **1. 행동 분류 모델**

- **파일**: `behavior_classification.py`
- **기능**: 사용자의 대화 속에서 행동 유형을 분류하여 갈등의 구체적인 원인을 파악
- **분류 카테고리**: 경쟁형/회피형/수용형/타협형/협력형
- **config**:
  - MAX_LENGTH: 256
  - BATCH_SIZE: 16
  - EPOCHS: 10
  - LEARNING_RATE: 2e-5
  - NUM_LABELS: 5
  - PATIENCE: 5
  - WEIGHT_DECAY: 0.01
- **text augmentation**
  - EDA
    - 단어 삭제, 교체, 추가, 순서 변경
  - 유사 임베딩 대체
    - 단어 임베딩 유사도 기반 대체
  - Contetualized Embedding 대체
    - 문맥 기반 단어 대체
- **📂 데이터셋 예시**:
  - 총 3,869개 (경쟁형:771개/회피형:759개/수용형:786개/타협형:767개/협력형:786개) -> 증강 후 21665개
문맥 및 감정 점수 모델에서 사용된 데이터셋(`behavior_dataset.json`)의 일부 예시는 다음과 같습니다:

```json
{
    "behavior_examples": [
        {
            "category": "경쟁형",
            "examples": [
                {"text": "이게 다 네가 제대로 준비 안 해서 그런 거 아니야?", "label": "경쟁형"},
                {"text": "네가 이렇게 할 줄 알았어", "label": "경쟁형"},
                {"text": "이런 기본적인 것도 못하면서 뭘 하겠다는 거야", "label": "경쟁형"}
            ]
        },
        {
            "category": "협력형",
            "examples": [
                {"text": "우리 이 문제를 함께 해결해 보자.", "label": "협력형"},
                {"text": "다음 번엔 내가 도와줄게.", "label": "협력형"},
                {"text": "우리가 힘을 합치면 잘 할 수 있을 거야.", "label": "협력형"}
            ]
        }
    ]
}
```
- **핵심 로직**:
  ```python
  def classify_behavior(input_text: str):
      """
      행동 유형을 분류하는 함수
      Args:
          input_text (str): 입력 텍스트
      Returns:
          prediction (str): 행동 유형 예측
      """
      model = torch.load("Behavior_classifier.pt")
      tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
      tokens = tokenizer(input_text, return_tensors="pt")
      output = model(**tokens)
      prediction = torch.argmax(output.logits, dim=-1)
      return prediction
  ```
  - **출력**: 행동 분류 결과 (ex. 경쟁형)
---

### **2. 문맥 및 감정 점수 모델**

- **파일**: `BERTbasedcontext.py`
- **기능**: 대화의 문맥과 감정을 분석하여 상황의 심각성을 점수화
- **text augmentation**
  - Random Insertion
    - 임의의 단어를 문장 내에 삽입하여 문장의 다양성 증가
  - Add Noise
    - 문자 수준에서 노이즈를 추가해 모델의 견고성 향상
- **📂 데이터셋 예시**:
  - 총 943개 -> 증강 후 2829개
문맥 및 감정 점수 모델에서 사용된 데이터셋(`BERT-based_dataset.json`)의 일부 예시는 다음과 같습니다:

```json
{
    "conversation_id": "conv_001",
    "utterances": [
        "여보세요? 방금 면접 결과 연락 받았어요!",
        "헐.. 떨려서 심장이 터질 것 같아요. 근데 최종 합격이래요!!",
        "아... 근데 연봉이 생각보다 너무 낮게 제시됐어요."
    ],
    "emotions": [0.6, 0.9, 0.3],
    "context_labels": {
        "situation": "job_interview_result",
        "emotion_flow": "anticipation->joy->disappointment"
    }
}
```
- **핵심 로직**:
```python
 def analyze_emotion(input_text: str):
    """
    입력 텍스트의 감정 점수를 분석
    Args:
        input_text (str): 입력 텍스트
    Returns:
        emotion_score (float): 감정 점수
    """
    model = torch.load("BERTbasedemotion_model.pt")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(input_text, return_tensors="pt")
    output = model(**tokens)
    emotion_score = torch.softmax(output.logits, dim=-1)[0, 1].item()
    return emotion_score
```
 - **출력**: 감정 점수 (0-1 범위)
---

### **3. 공감 점수 측정 모델**

- **파일**: 'empathy_data_preprocessig.py', `empathy_score.py.py`
- **기능**: 대화에서 상대방에 대한 공감을 점수화하여 협력 가능성 평가
- **text augmentation**
  - Random deletion
    - 문장에서 임의의 단어 삭제
  - Random swap
    - 문장 내 임의의 두단어의 위치를 교환하여 문장의 구조 변형
  - Random insertion
    - 문장 내 임의의 위치에 단어 삽입하여 다양성 증가시킴
  - Synonym replacement
    - init에서 한국어 단어와 유사어 리스트를 사전 형태로 저장함
    - 단순 사전 기반으로 단어를 유사어로 치환하여 문장의 의미 보존해 다양성 증가
  - Noise Injection
    - 텍스트에 노이즈 주입해 견고성 향상시킴
  - Augment text
    - 여러 증강 기법 조합해 텍스트 변형
- **📂 데이터셋 예시**:
 - 원본 데이터 크기: 2,692개 대화 -> 증강 후 데이터 크기: 14787
문맥 및 감정 점수 모델에서 사용된 데이터셋(`empathy_dataset.json`)의 일부 예시는 다음과 같습니다:

```json
{
    "dialogue_id": "001",
    "utterances": [
        {
            "utterance_id": "001_1",
            "speaker": "A",
            "text": "자기야, 오늘 기분 어때?",
            "emotion": "중립",
            "empathy_score": 0.0
        },
        {
            "utterance_id": "001_2",
            "speaker": "B",
            "text": "회사에서 실수해서 팀장님한테 혼났어... 많이 속상해",
            "emotion": "슬픔",
            "empathy_score": 0.0
        },
        {
            "utterance_id": "001_3",
            "speaker": "A",
            "text": "많이 힘들었겠다... 우리 자기가 평소에 얼마나 열심히 하는데. 저녁에 맛있는 거 먹으러 갈까?",
            "emotion": "공감",
            "empathy_score": 0.9
        }
    ]
}
```
- **핵심 로직**:
```python
def compute_empathy_score(dialogues: List[str]):
    """
    대화의 공감 점수를 계산
    Args:
        dialogues (List[str]): 대화 리스트
    Returns:
        empathy_score (float): 공감 점수
    """
    model = torch.load("bestmodel.pt")
    features = preprocess_dialogues(dialogues)
    output = model(features)
    empathy_score = torch.mean(output.logits).item()
    return empathy_score
```
 - **출력**: 공감 점수 (0-1 범위)
---

### **4. 모델 통합 및 과실 비율 산출**

- **파일**: 'score_multi.py'
- **기능**: 3개의 모델의 결과를 통합하여 최종 과실 비율 산출
- **핵심 로직**: 상황 분석 및 요약 -> 행동 분류 -> 문맥 감정 -> 공감 점수 -> 최종 판결문 출력 
- **상황 분석** (SituationAnalyzer)
  - 기능: 갈등 텍스트를 분석하여 주요 상황과 각 상황의 중요도를 계산하고 관련 문장 추출
  - 주요 갈등 상황과 중요도 분석
    - 관련 문장 빈도(frequency_score): 문장에서 키워드와 일치하는 비율
    - 감정 중요도(emotion_importance): 관련 문장의 감정 점수 평균ㄴ
```python
class SituationAnalyzer:
    async def analyze_text(self, text: str) -> Dict[str, Dict[str, any]]:
        """
        텍스트에서 주요 갈등 상황과 중요도를 분석합니다.
        Args:
            text (str): 입력 텍스트
        Returns:
            Dict: 상황별 중요도와 관련 문장
        """
        prompt = f"""Analyze the following text:
        Text: {text}

        Please respond in the following JSON format:
        {{
            "situations": [
                {{
                    "category": "situation category name",
                    "importance": importance of the situation (0 to 1),
                    "related_sentences": ["sentence 1", "sentence 2"]
                }},
                ...
            ]
        }}"""

        response = await self.client.chat.completions.create(
            model=Config.GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing conflict situations."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = json.loads(response.choices[0].message.content)
        return {
            s["category"]: {
                "importance": s["importance"],
                "sentences": s["related_sentences"]
            } for s in analysis["situations"]
        }

```

- **행동 분류 및 점수 매핑 (Behavior_classification)**
  - 클래스: CustomBERTClassifier
  - 행동 점수 매핑
      ```python
      def map_category_score(category):
    score_map = {
        "경쟁": 0,
        "회피": 0,
        "타협": 0.5,
        "협력": 1,
        "수용": 1
    }
    return score_map.get(category, 0)
    ```
- **감정 분석 (Emotion Analysis)**
  - 클래스: SentenceEmotionAnalyzer
- **과실 비율 계산 (Fault Ratio Calculation)**
  - ConflictAnalyzer
  - 기능: 감정 점수, 행동 점수, 상황 중요도를 기반으로 원고와 피고의 과실 비율을 계산
  - 핵심 로직
    ```python
    async def analyze_content(self, content: str) -> dict:
    """
    텍스트 데이터를 분석하고 과실 비율을 계산합니다.
    Args:
        content (str): 분석할 텍스트
    Returns:
        dict: 분석 결과
    """
    situation_data = await self.situation_analyzer.analyze_text(content)
    speaker_scores = {"plaintiff": 0, "defendant": 0}

    for situation, data in situation_data.items():
        sentences = data["sentences"]
        importance = data["importance"]

        emotion_scores = self.emotion_analyzer.analyze_sentences(sentences)
        behavior_scores = predict_category_scores(
            sentences, self.behavior_model, self.tokenizer, self.label_map, self.device
        )

        for e_score, b_score in zip(emotion_scores, behavior_scores):
            score = e_score * b_score * importance
            speaker_scores["plaintiff" if "내가" in sentences else "defendant"] += score

    total_score = sum(speaker_scores.values())
    fault_rate = round((speaker_scores["plaintiff"] / total_score) * 100, 2) if total_score else 50.0
    return {"plaintiff_fault_rate": fault_rate}
    ```

---
## 🎙️ STT 및 OCR 데이터 처리

### **1. STT (Speech-to-Text)**

- **파일**: 'audio_process.py', 'download_s3_file.py'
- **사용된 라이브러리**: OpenAI Whisper, Boto3
- **엔드 포인트**: /speech-to-text
- **기능**: S3에 저장된 음성 파일을 다운로드하여 텍스트로 변환
- **핵심 로직**:
 ```python
  async def download_s3_file(url):
      """
      AWS S3에서 파일을 다운로드합니다.
      Args:
          url (str): S3 객체 URL
      Returns:
          str: 다운로드된 파일의 경로
      """
      bucket_name, object_key, filename = await parse_s3_url(url)
      temp_dir = "temp"
      os.makedirs(temp_dir, exist_ok=True)

      s3_client.download_file(bucket_name, object_key, temp_file_path)
      if os.path.exists(temp_file_path):
          logger.info(f"File downloaded successfully: {temp_file_path}")
      else:
          raise FileNotFoundError("Downloaded file not found")
      return temp_file_path

 async def transcribe_audio(file_path):
    """
    음성 파일을 텍스트로 변환합니다.
    Args:
        file_path (str): 음성 파일 경로
    Returns:
        str: 변환된 텍스트
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text
 ```

### **2. OCR (Optical Character Recognition)**

- **파일**: 'audio_process.py', 'download_s3_file.py'
- **사용된 라이브러리**: EasyOCR, OpenCV
- **엔드 포인트**: /image-to-text
- **기능**: S3에 저장된 음성 파일을 다운로드하여 텍스트로 변환.
- **핵심 로직**:
  ```python
  async def extract_text_from_image(image_path):
      """
      주어진 이미지에서 텍스트를 추출합니다.
      Args:
          image_path (str): 이미지 파일 경로
      Returns:
          list[dict]: 추출된 텍스트와 확률 정보가 포함된 리스트
      """
      # EasyOCR 리더 객체 생성
      reader = easyocr.Reader(['ko', 'en'])
      
      # 이미지 파일 읽기
      image = cv2.imread(image_path)
      if image is None:
          raise FileNotFoundError(f"Image file not found: {image_path}")

      # 텍스트 추출
      results = reader.readtext(image)

      # 결과 포맷팅
      extracted_texts = []
      for bbox, text, prob in results:
          extracted_texts.append({
              "text": text,
              "probability": prob,
              "bounding_box": bbox
          })
      return extracted_texts
    ```
---
## Async Processing in Mediation Service

### **주요 비동기 처리 흐름**
- **RabbitMQ 기반 비동기 처리**
  - 기능: RabbitMQ 메시지 큐를 사용하여 AI 모델을 호출하고 결과 처리
  - 핵심 로직:
  ```python
  def process_message(ch, method, properties, body):
    """
    RabbitMQ 메시지 처리 콜백 함수.
    Args:
        ch: 채널
        method: 메시지 전달 정보
        properties: 메시지 속성
        body: 메시지 본문
    """
    try:
        message = json.loads(body.decode('utf-8'))
        content = message.get("content")
        request_id = message.get("id")

        if not content or not request_id:
            raise ValueError("Invalid message: Missing 'content' or 'id'")

        execute_score_multi_and_callback(content, request_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
  ```

- **BackgroundTasks 처리**
  - 기능: FastAPI BackgroundTasks를 사용하여 비동기적으로 결과를 처리하고 콜백 URL에 POST 요청
  - 핵심 로직:
```python
  @router.post("/judgement", status_code=202)
async def process_judge(request: JudgeRequest, background_tasks: BackgroundTasks):
    """
    요청 데이터를 비동기적으로 처리하고 202 Accepted 응답 반환.
    Args:
        request (JudgeRequest): 요청 데이터
        background_tasks (BackgroundTasks): 백그라운드 작업 관리 객체
    Returns:
        dict: 응답 상태
    """
    background_tasks.add_task(execute_test_response_and_callback, request.content, request.id)
    return {"status": "accepted", "message": "Judgement processing started."}
```

---
## 🛠️사용 기술 
- AI 및 NLP
  - OpenAI GPT-4
  - Hugging Face Transformers
    - BERT 기반 감정 분석 모델
    - BERT 기반 행동 분류 모델
  - PyTorch
- 음성 및 이미지 처리
  - OpenAI Whisper
  - EasyOCR
- 백엔드 및 API
  - FastAPI
    - 고성능 비동기 API 서버
  - HTTPx
    - 비동기 HTTP client로 외부 API 통신
  - pika
    - RabbitMQ 메세지 큐 관리 및 작업 분배
  - 데이터 관리 및 클라우드
    - AWS S3
      - 음성 및 이미지 데이터 저장 및 관리
## 📮 문의

- 팀 GitHub: [KTB16Team](https://github.com/KTB16Team)
- 관련 문의는 Issues를 통해 남겨주세요.
