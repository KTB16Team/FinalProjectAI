import os
import easyocr
import cv2
from services.download_s3_file import download_s3_file
async def extract_text_from_image(image_path):
    """
    주어진 이미지에서 텍스트를 추출합니다.
    Args:
        image_path (str): 이미지 파일 경로
    Returns:
        list[dict]: 추출된 텍스트와 확률 정보가 포함된 리스트
    """
    try:
        # EasyOCR 리더 객체 생성
        reader = easyocr.Reader(['ko', 'en'])

        # 이미지 파일 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 이미지에서 텍스트 추출
        results = reader.readtext(image)

        # 결과를 리스트로 정리
        extracted_texts = []
        for bbox, text, prob in results:
            extracted_texts.append({
                "text": text,
                "probability": prob,
                "bounding_box": bbox
            })
        processed_messages = classify_messages(extracted_texts)
        return processed_messages

    except Exception as e:
        print(f"Error extracting text from image: {e}")
        raise

def classify_messages(data):
    """
    메시지를 좌측 정렬(상대방)과 우측 정렬(사용자)으로 분류하고, 사용자 정보를 추가하며 반복되는 텍스트를 인식하고 자동으로 추출하여 제거합니다.
    """
    formatted_messages = []
    current_user = None
    current_time = None
    temp_message = ""
    previous_text = None  # 이전 텍스트 기록
    previous_user = None  # 이전 사용자의 이름 기록
    repeated_names = set()  # 반복되는 이름이나 텍스트 집합
    time_updated = False  # 시간이 업데이트되었는지 확인
    first_opponent_message_added = False  # 첫 상대방 메시지를 추가했는지 확인

    for entry in data:
        text = entry['text']
        bounding_box = entry['bounding_box']

        # bounding_box의 X 좌표 추출 (첫 번째 좌표의 X 값)
        x_start = bounding_box[0][0]

        # 사용자 이름 추출 또는 기본값 설정
        if x_start < 400:
            user = "상대방"
            if not first_opponent_message_added:
                repeated_names.add(text)
                first_opponent_message_added = True
                continue
        else:
            user = "사용자"

        # 시간 추출 (시간 정보가 텍스트에 포함된 경우 처리)
        if "오후" in text or "오전" in text:
            current_time = text
            time_updated = True
            continue

        # 시간 업데이트 이후 첫 번째 메시지이면서 좌측 정렬인 경우, 반복되는 이름으로 추정
        if time_updated and user == "상대방":
            repeated_names.add(text)
            time_updated = False
            continue

        # 반복되는 이름 제거
        if text in repeated_names:
            continue

        # 같은 사용자가 말한 경우 메시지를 합침
        if current_user == user and current_time:
            temp_message += f" {text}".strip()
        else:
            # 새로운 사용자로 변경되거나 시간이 없을 경우 메시지 추가
            if temp_message:
                formatted_messages.append(f"{current_time}, {current_user} : {temp_message.strip()}")
                temp_message = ""

            current_user = user
            temp_message = text

        # 이전 사용자와 텍스트 기록 업데이트
        previous_user = user
        previous_text = text

    # 마지막 메시지 추가
    if temp_message:
        formatted_messages.append(f"{current_time}, {current_user} : {temp_message.strip()}")

    return "\n".join(formatted_messages)



async def process_image_file(url):
    """
    주어진 URL의 이미지를 다운로드하여 텍스트를 추출합니다.
    Args:
        url (str): 이미지 URL
    Returns:
        list[dict]: 추출된 텍스트와 확률 정보가 포함된 리스트
    """
    temp_file_path = None
    try:
        # 이미지 파일 다운로드
        temp_file_path = await download_s3_file(url)

        # 이미지에서 텍스트 추출
        extracted_text = await extract_text_from_image(temp_file_path)
        return extracted_text

    except Exception as e:
        print(f"Error processing image file: {e}")
        raise

    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)