import torch
import unittest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..services.empathy_score import DialogueEmpathyModel, EmpathyLoss, train_model, eval_model
from ..services.empathy_data_preprocessing import DialogueDataset, collate_fn, split_data

def mock_dataset():
    """테스트를 위한 더미 데이터셋 생성"""
    return [
        {
            "dialogue_id": "001",
            "utterances": [
                {"utterance_id": "001_1", "text": "자기야, 오늘 기분 어때?", "speaker": "A", "empathy_score": 0.0},
                {"utterance_id": "001_2", "text": "회사에서 실수해서 팀장님한테 혼났어... 많이 속상해", "speaker": "B", "empathy_score": 0.0},
                {"utterance_id": "001_3", "text": "많이 힘들었겠다... 우리 자기가 평소에 얼마나 열심히 하는데. 저녁에 맛있는 거 먹으러 갈까?", "speaker": "A", "empathy_score": 0.9}
            ]
        },
        {
            "dialogue_id": "002",
            "utterances": [
                {"utterance_id": "002_1", "text": "나 요즘 살이 자꾸 찌는 것 같아서 너무 스트레스 받아...", "speaker": "A", "empathy_score": 0.0},
                {"utterance_id": "002_2", "text": "우리 자기 충분히 예쁜데! 그래도 자기가 신경 쓰이면 우리 같이 운동해볼까? 나도 요즘 운동하고 싶었거든!", "speaker": "B", "empathy_score": 0.85},
                {"utterance_id": "002_3", "text": "같이 가주는 거야? 역시 자기밖에 없다ㅠㅠ", "speaker": "A", "empathy_score": 0.0}
            ]
        },
        {
            "dialogue_id": "003",
            "utterances": [
                {"utterance_id": "003_1", "text": "자기야... 우리 부모님이 또 결혼 얘기 꺼내셨어", "speaker": "A", "empathy_score": 0.0},
                {"utterance_id": "003_2", "text": "많이 부담됐겠다... 우리 천천히 준비하면 되는데. 오늘 저녁에 만나서 얘기 나눌까?", "speaker": "B", "empathy_score": 0.95}
            ]
        }
    ]

class TestDialogueEmpathyModel(unittest.TestCase):
    def setUp(self):
        """테스트 세팅 초기화"""
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 저장된 모델 불러오기
        self.model = DialogueEmpathyModel(
            input_size=768,
            hidden_size=256,
            num_speakers=10,
            num_layers=1,
            dropout=0.5
        ).to(self.device)

        self.model.load_state_dict(torch.load("Final/best_model.pt", map_location=self.device))
        self.model.eval()

    def test_dataset_loading(self):
        """Dataset 로드 및 토크나이저 작동 확인"""
        dataset = DialogueDataset(mock_dataset(), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        for batch in dataloader:
            utterances, speaker_ids, empathy_scores = batch
            self.assertEqual(len(utterances), 2)
            self.assertEqual(utterances.size(-1), 768)  # BERT hidden size 확인
            break

    def test_model_prediction(self):
        """저장된 모델을 활용한 예측 테스트"""
        dataset = DialogueDataset(mock_dataset(), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        for batch in dataloader:
            utterances, speaker_ids, _ = batch
            utterances, speaker_ids = utterances.to(self.device), speaker_ids.to(self.device)

            with torch.no_grad():
                predictions = self.model(utterances, speaker_ids)

            self.assertIsNotNone(predictions, "Predictions should not be None")
            self.assertEqual(predictions.size(0), utterances.size(0), "Batch size should match predictions")
            break

if __name__ == "__main__":
    unittest.main()
