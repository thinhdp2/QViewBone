# Q-ViewBone

Mã thực nghiệm cross-view VQA, chạy trên máy cá nhân hoặc GPU Google Colab. Dự án ưu tiên đo lợi ích của ảnh thứ hai trước khi học chính sách STOP / REQUEST_VIEW.

Đánh giá chi tiết góp ý và các điều chỉnh nghiên cứu: [docs/review-roadmap.md](docs/review-roadmap.md).

## Có sẵn

- Đọc bộ JSONL / CSV hiện có, lập manifest ảnh cục bộ kèm SHA-256; không phụ thuộc HTML, Node.js, PowerShell hay workbook khi chạy thí nghiệm.
- Tạo bảng xác minh cặp ảnh, lọc category đã duyệt và gộp hai episode đảo chiều thành một cặp–câu hỏi.
- Chia tập theo bệnh nhân bằng seed cố định, ổn định khi đổi thứ tự dữ liệu hoặc thêm bệnh nhân.
- Sáu điều kiện: `A`, `B`, `AB`, `BA`, `A_to_B`, `B_to_A`.
- Adapter Qwen2.5-VL dùng ảnh gốc, lịch sử tuần tự lấy dự đoán của chính mô hình. Gold, chẩn đoán audit, segment và thông tin bệnh nhân không đưa vào prompt.
- Backend `dummy` để thử pipeline không cần GPU / model; đầu ra luôn gắn nhãn `dry_run`.
- Tự lưu mỗi prediction và chạy tiếp khi ngắt; chặn trộn cấu hình / dữ liệu / GPU / precision khác vào cùng run.
- Exact-set accuracy theo category, parse failure, số ảnh đầu vào, Rescue/Harm hai chiều, so sánh thứ tự, CI bootstrap theo bệnh nhân.

Chưa triển khai huấn luyện policy, confidence calibration, IoU ROI hoặc clinical grading cho chẩn đoán. Adapter VLM đã viết theo API chính thức nhưng chưa được chạy bằng GPU trong môi trường phát triển hiện tại. Kiểm thử `dummy` không chứng minh chất lượng VLM.

## Cấu trúc

```text
QViewBone/
  configs/experiment.toml
  notebooks/colab_pilot.ipynb
  src/qviewbone/
    settings.py       # đường dẫn, cấu hình và biến môi trường
    dataset.py        # manifest, xác minh cặp, task, chia tập
    prompts.py        # prompt không chứa gold
    inference.py      # dummy / Qwen2.5-VL, checkpoint, resume
    analysis.py       # chấm và phân tích utility
    cli.py
  tests/test_pipeline.py
  docs/review-roadmap.md
  data/               # sinh khi chạy, không đưa vào Git
  outputs/            # sinh khi chạy, không đưa vào Git
```

## Chạy trên máy cá nhân

Python 3.10 trở lên:

```bash
cd QViewBone
python -m pip install -e .
python -m qviewbone settings
python -m qviewbone prepare
```

Trong workspace hiện tại, tự tìm `../vqa-dataset` và ảnh trong `../vqa-dataset/image-cache`. Không dùng đường dẫn `G:\...` trong code.

Lệnh `prepare` tạo:

- `data/prepared/images_manifest.csv`: đường dẫn ảnh tương đối và hash.
- `data/prepared/pair_review.csv`: bản sao bảng ứng viên để người kiểm tra điền; không ghi đè nếu đã có.
- `data/prepared/eval_dataset.jsonl`: một dòng cho một cặp–category.
- `data/prepared/patient_splits.json` và `preparation_report.json`.

Hiện chưa có cặp ảnh được xác minh nên lần chuẩn bị mặc định xuất **0 task hợp lệ**. Đây là trạng thái đúng, không phải lỗi thiếu ảnh.

Để thử plumbing bằng ứng viên chưa xác minh:

```bash
python -m qviewbone prepare --allow-unverified --max-pairs 3
python -m qviewbone run --backend dummy --run-name smoke --allow-unverified --limit 3
python -m qviewbone analyze outputs/smoke
```

Chạy lại lệnh `run` giống hệt để resume. Kết quả `dummy` là dữ liệu giả để kiểm tra code, không dùng làm kết quả nghiên cứu. Sau smoke test hãy chạy lại `prepare` không có `--allow-unverified` để trở về dữ liệu đã duyệt.

## Xác minh cặp ảnh

Điền vào `pair_review.csv`:

- `same_examination_verified=True`, `pair_confirmed_same_lesion=True` khi đã xác minh bằng ảnh / hồ sơ.
- `projection_A`, `projection_B` nếu xác định được; không tự suy ra tư thế từ thứ tự ảnh.
- `approved_categories`, ví dụ `C1.1;C1.2;C2.1`: chỉ những câu hỏi đã kiểm tra vùng đích và gold có thể trả lời từ ảnh. Chỉ xác nhận quan hệ cặp chưa đủ xác nhận mọi câu hỏi.
- `status=verified` hoặc `approved`; có thể ghi người duyệt, thời điểm và ghi chú.

Không sửa các cột định danh cặp. Không tự gán xác minh từ việc hai ảnh có cùng gold. Sau đó chạy `prepare` lại. Mặc định lấy tối đa 30 cặp đủ điều kiện; `--max-pairs 0` lấy toàn bộ.

## Colab và Git

Mở [notebooks/colab_pilot.ipynb](notebooks/colab_pilot.ipynb), chọn runtime GPU nếu chạy Qwen. Notebook có form để nhập Git URL / ref, thư mục Drive, model, pixel budget và tên run.

1. Đưa **nội dung thư mục `QViewBone`** thành root của một repository. Notebook cũng nhận ra trường hợp thư mục này nằm bên trong repository lớn hơn.
2. Đặt bộ JSONL/CSV và `image-cache/` trên Drive, ví dụ `MyDrive/QViewBone/vqa-dataset/`.
3. Đổi `REPO_URL` trong notebook sang URL thật. Chưa có remote được cấu hình hoặc push từ workspace này.
4. `BACKEND="dummy"` để kiểm tra trước; đổi `BACKEND="qwen"` khi muốn cài dependencies VLM và chạy model thật.
5. Lưu `prepared`, `pair_review.csv`, outputs trên Drive. Có thể copy cache ảnh xuống `/content` để đọc nhanh; đây là bản sao, không di chuyển ảnh gốc.

Code được quản lý trên Git; dataset, workbook, signed URL, model weights và kết quả thực nghiệm không nằm trong repository này. Nếu repo riêng tư, dùng cơ chế đăng nhập Git / Colab Secrets, không nhúng token vào notebook hoặc URL commit.

Theo [Colab FAQ](https://research.google.com/colaboratory/faq.html), phần cứng và thời lượng runtime có thể thay đổi; runtime tạm thời không thay thế lưu trữ bền vững. GPU nào có sẵn sẽ được phát hiện lúc chạy, không mặc định luôn có T4/A100.

## Cấu hình động

Ưu tiên: **tham số API/CLI > biến môi trường > TOML > tự phát hiện đường dẫn**. CLI override đường dẫn đặt trước subcommand, ví dụ:

```bash
python -m qviewbone --data-dir /content/drive/MyDrive/QViewBone/vqa-dataset settings
```

Trong Colab:

```python
import os
os.environ['QVB_DATA_DIR'] = '/content/drive/MyDrive/QViewBone/vqa-dataset'
os.environ['QVB_IMAGE_DIR'] = '/content/qviewbone-images'
os.environ['QVB_PREPARED_DIR'] = '/content/drive/MyDrive/QViewBone/prepared'
os.environ['QVB_PAIR_REVIEW'] = '/content/drive/MyDrive/QViewBone/pair_review.csv'
os.environ['QVB_OUTPUT_DIR'] = '/content/drive/MyDrive/QViewBone/runs'
os.environ['QVB_MODEL_ID'] = 'Qwen/Qwen2.5-VL-3B-Instruct'
os.environ['QVB_DEVICE'] = 'auto'
os.environ['QVB_DTYPE'] = 'auto'
os.environ['QVB_QUANTIZATION'] = 'none'
os.environ['QVB_MAX_PIXELS'] = '401408'
```

Các đường dẫn tương đối trong settings được tính từ root project (`QVB_ROOT`), không phụ thuộc thư mục đang đứng. Đường dẫn ảnh trong task luôn tương đối với `QVB_IMAGE_DIR`, nên chuyển Windows → Drive → `/content` không cần sửa từng task.

`device=auto` dùng CUDA nếu có, nếu không dùng CPU. `dtype=auto` chọn bfloat16 khi GPU hỗ trợ, float16 cho CUDA còn lại, float32 trên CPU. Có lựa chọn NF4 4-bit khi CUDA khả dụng; không đổi precision/pixel budget giữa các điều kiện trong cùng run. Nếu đổi GPU/precision/config, dùng tên run mới để giữ thí nghiệm nhất quán.

Adapter hiện hỗ trợ **họ Qwen2.5-VL**: model ID có thể đổi 3B ↔ 7B hoặc đường dẫn checkpoint cùng kiến trúc. Model họ khác cần adapter riêng, không chỉ thay tên. Mặc định là 3B để thử pipeline, chưa coi đó là một VLM y khoa đã được xác nhận. Theo [tài liệu Qwen2.5-VL](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl), ngân sách pixel có thể cấu hình; cần kiểm tra lại hiệu quả khi giảm độ phân giải X-quang. Phiên bản Transformers được ghim để tránh tự đổi API khi cài lại Colab.

## Chạy model thật

```bash
python -m pip install -e '.[vlm]'
python -m qviewbone run --backend qwen --run-name qwen25-3b-pilot --split train
python -m qviewbone analyze outputs/qwen25-3b-pilot
```

Chỉ dùng `--allow-unverified` khi chủ động chạy thăm dò và ghi rõ giới hạn này trong báo cáo. Chưa xác minh hết không nên dùng kết quả làm benchmark chính thức.

`run.json` ghi settings, hash task, source inference/prompt, phiên bản thư viện, GPU/dtype và model commit. `predictions.jsonl` lưu sau mỗi output. Nếu Colab ngắt giữa dòng cuối, lệnh resume loại phần ghi dở rồi tính tiếp; lỗi ở giữa file sẽ được báo. Không chạy hai tiến trình ghi cùng một run.

## Đọc kết quả

- `metrics.csv`: exact-set accuracy, parse failure, số ảnh/condition.
- `view_utility.csv`: đúng trước/sau, delta và Rescue/Harm/Preserved/Unresolved.
- `order_effects.csv`: AB so với BA, tuần tự hai chiều.
- `summary.json`: loại kết quả (`dry_run`, `exploratory`, `verified_pairs`), CI và phương pháp chấm.

Scorer so sánh tập đáp án sau chuẩn hóa Unicode, khoảng trắng và chữ hoa/thường; không dùng substring, không bỏ dấu phủ định. Đây là lexical matching, không phải thẩm định y khoa. Với từ đồng nghĩa có thể cung cấp `analyze ... --aliases aliases.json` (object từ chuỗi chuẩn hóa sang nhãn chuẩn). Khóa bảng alias trên train/val trước khi chấm test; không sửa theo câu trả lời test. C2.2 cần thêm quy trình adjudication. ROI không dùng chung gold giữa hai ảnh và chưa được runner này chấm IoU.

`mean_views` đếm ảnh được mở (1 hoặc 2), không đo latency/token/FLOPs. Trình tự hai lượt có thêm chi phí xử lý lịch sử, dù đều mở hai ảnh. CI lấy mẫu lại theo bệnh nhân và giữ các task cùng bệnh nhân đi cùng nhau; 1.794 episode không được xem là 1.794 quan sát độc lập.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Test bao phủ chặn cặp chưa duyệt, duyệt từng category, gộp episode đảo chiều, lịch sử không chứa gold, resume, file ghi dở, ảnh bị đổi, ổn định split và scoring có phủ định. CI chạy các test không cần dataset/GPU. Trước thí nghiệm chính thức cần chạy smoke test VLM trên GPU Colab và kiểm tra prompt/ảnh thực tế.
