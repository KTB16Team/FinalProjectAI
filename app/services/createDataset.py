import json
import random

# 경쟁형 카테고리의 다양한 예시 텍스트 (SNS 대화 스타일에 맞게 간결하고 직설적으로 표현)
competition_examples = [
    "네가 남사친이랑 그렇게 자주 만나는 게 난 정말 불편해. 이번엔 내 말대로 거리를 좀 두자.",
    "왜 또 여사친이랑 저녁을 먹은 거야? 내가 그렇게 싫다고 했는데도 네가 무시한 거잖아.",
    "내가 남사친 문제로 그렇게 얘기했는데 왜 자꾸 나를 무시해? 이번엔 내 말 좀 들어줘.",
    "여사친이랑 늦게까지 얘기하는 게 나는 진짜 이해가 안 돼. 이번엔 내 말대로 해줬으면 좋겠어.",
    "네가 남사친 만나는 걸 그렇게 허락해줬더니 이번엔 또 영화까지 본다고? 내 입장도 생각해줘.",
    "여사친한테 그렇게 신경 쓸 거면 나한테는 왜 신경 안 써? 이번엔 내 부탁 좀 들어줘.",
    "네가 남사친이랑 나보다 더 자주 얘기하는 것 같아. 이번엔 그만 좀 했으면 좋겠어.",
    "내가 싫다고 했는데도 자꾸 여사친 문제로 나를 시험하는 것 같아. 이번엔 내 의견을 따라줘.",
    "남사친이랑 만나는 건 좋지만 그 시간을 나한테도 써줬으면 좋겠어. 이번엔 내 말 좀 들어줘.",
    "여사친 문제로 자꾸 싸우고 싶지 않아. 이번엔 네가 좀 거리를 두면 좋겠어.",
    "네가 남사친이랑 너무 가깝게 지내는 게 난 정말 힘들어. 내 말도 좀 들어줄 수 없어?",
    "여사친이랑 시간을 보내는 건 이해하지만, 이번엔 나랑 더 많은 시간을 보냈으면 좋겠어.",
    "남사친과 여행 가겠다는 건 정말 아니지 않아? 이번엔 나를 생각해서 그만둬줬으면 좋겠어.",
    "네가 여사친이랑 그렇게 다정하게 메시지를 주고받는 게 너무 불편해. 내 입장도 생각해줘.",
    "남사친과의 관계가 왜 그렇게 중요한지 모르겠어. 이번엔 내 말대로 거리를 좀 두자.",
    "내가 여사친 문제로 서운해하는 걸 알면서도 자꾸 그러는 거, 정말 나를 힘들게 해. 이번엔 내 말을 들어줘.",
    "남사친이랑의 관계가 친구라 해도 나는 너무 불편해. 네가 내 입장을 좀 고려해줬으면 좋겠어.",
    "여사친이랑 시간을 보내는 걸 내가 반대하는 이유를 네가 이해해줬으면 좋겠어. 이번엔 내 말 좀 따라줘.",
    "남사친 문제로 우리가 이렇게 자주 싸우는 거 정말 싫어. 이번엔 네가 좀 이해해줬으면 좋겠어.",
    "여사친과의 모임에 나를 제외하는 건 너무해. 이번엔 같이 가는 걸로 해줘.",
    "남사친이랑 단둘이 술 마시는 건 정말 아니다. 내 말을 좀 들어줄 수 없어?",
    "네가 여사친 문제로 내가 힘들어하는 걸 알잖아. 이번엔 나를 위해 그만해 줘.",
    "남사친이랑 자주 만나는 것 때문에 나도 많이 불안해. 이번엔 나를 이해해줘.",
    "여사친이랑 주말마다 시간을 보내는 건 정말 과해. 이번 주말은 나랑 보내자.",
    "네가 남사친이랑 시간을 보내는 건 이해하지만, 내가 이렇게 불편해하는 것도 이해해 줘.",
    "여사친한테 선물을 준다고? 내 입장에선 그게 너무 불편해. 이번엔 나를 생각해줘.",
    "남사친에게 그렇게 친절하게 대하는 거 나도 신경 쓰여. 이번엔 내 입장을 존중해줘.",
    "네가 여사친이랑 너무 가까운 것 같아. 나도 네가 나한테만 집중해줬으면 좋겠어.",
    "남사친이랑의 관계 때문에 우리가 이렇게 싸우는 건 정말 피곤해. 이번엔 내 말 좀 들어봐.",
    "여사친이랑 약속 잡기 전에 나한테 물어봐 줬으면 좋겠어. 이번엔 내 말대로 해줘.",
    "남사친과의 관계는 친구라고 해도 난 불안해. 내 감정을 조금이라도 고려해줬으면 좋겠어.",
    "여사친이랑 연락을 계속하는 게 나를 불안하게 해. 이번엔 조금 줄여줬으면 해.",
    "네가 남사친과 자꾸 만나는 걸 나는 이해할 수 없어. 내 입장을 고려해줘.",
    "여사친과의 대화에서 나를 비밀로 하는 게 너무 싫어. 이번엔 내 말을 들어줘.",
    "남사친이랑 여행 가는 거 나한테 얘기도 없이 결정한 거 너무해. 이번엔 취소하자.",
    "여사친이랑 함께 있는 사진을 왜 그렇게 자주 올리는 거야? 나한테 신경 써줬으면 좋겠어.",
    "남사친이랑 밤 늦게까지 연락하는 거 나 정말 싫어. 이번엔 내 말을 들어줄래?",
    "여사친 문제로 내가 계속 불편하다고 했잖아. 이번엔 그만 좀 해줘.",
    "남사친이랑 연락하는 게 친구라 해도 나한테는 정말 스트레스야. 이번엔 나를 위해 줄여줘.",
    "여사친과의 관계가 친구라지만, 내 입장에선 너무 힘들어. 이번엔 내 말 좀 들어줘.",
    "남사친과의 만남은 줄여줬으면 좋겠어. 나도 네가 나한테 더 집중해줬으면 해.",
    "여사친이랑 그렇게 자주 연락하는 건 정말 과한 것 같아. 이번엔 나랑 시간을 더 보내줘.",
    "남사친이랑 저녁을 같이 먹는 게 나한테는 큰 부담이야. 내 감정을 존중해줘.",
    "여사친과 나보다 더 많은 시간을 보내는 게 너무 서운해. 이번엔 내 입장을 생각해줘.",
    "네가 남사친에게 신경 쓰는 것보다 나에게 더 신경 써줬으면 좋겠어.",
    "여사친과 단둘이 만나는 걸 멈춰줬으면 해. 나를 위해서라도.",
    "남사친 문제로 계속 싸우는 거 정말 지겨워. 이번엔 네가 양보해줘.",
    "여사친이랑의 관계를 내가 받아들이기엔 너무 어려워. 내 감정도 좀 생각해줘."

]

# 경쟁형 카테고리의 데이터셋 형식으로 최대한 많이 생성 (중복 없는 다양한 문장)
data = [{"text": text, "label": "경쟁형"} for text in competition_examples]

# JSON 파일로 저장 (띄어쓰기 및 줄바꿈 없이)
with open("competition_data_dense.json", "w", encoding="utf-8") as f:
    json_line = json.dump(data, f, ensure_ascii=False)
    f.write(json_line + ",\n")

print("경쟁형 데이터셋 생성 완료: competition_data_dense.json")