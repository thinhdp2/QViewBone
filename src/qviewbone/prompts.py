SYSTEM = ('Bạn đang thực hiện thí nghiệm VQA trên ảnh X-quang. Chỉ dùng ảnh được cung cấp. '
          'Trả lời ngắn gọn bằng tiếng Việt. Xuất đúng JSON {"answer": ["giá trị 1", "giá trị 2"]}. '
          'Nếu không đủ bằng chứng, trả lời {"answer": ["không xác định"]}.')
PROMPT_VERSION = "v1"


def first_turn(question, image_paths):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "path": str(p)} for p in image_paths]
             + [{"type": "text", "text": question}]}]


def sequential(question, first_image, second_image, initial_prediction):
    # Free-running history: the assistant's raw prediction, never gold.
    return first_turn(question, [first_image]) + [
        {"role": "assistant", "content": [{"type": "text", "text": initial_prediction}]},
        {"role": "user", "content": [{"type": "image", "path": str(second_image)},
         {"type": "text", "text": f"Ảnh bổ sung đã được cung cấp. Cập nhật câu trả lời cho cùng câu hỏi: {question}"}]}]
