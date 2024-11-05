class PromptingService:
    def __init__(self):
        pass

    def Acting(self):
        return "Acting"

    def Emotional(self):
        return "Emotional"

    def Situational(self):
        return "Situational"

    def __str__(self):
        # dir(self)를 사용해 클래스의 모든 메서드 이름을 가져오고, 특수 메서드를 제외한 사용자 정의 메서드만 필터링
        methods = [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith("__")]
        return f"This is a PromptingService instance with methods: {', '.join(methods)}"

# 사용 예시
service = PromptingService()
print(service)  # 출력: This is a PromptingService instance with methods: Acting, Emotional, Situational