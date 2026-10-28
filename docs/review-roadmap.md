# Nhận xét lộ trình triển khai

**Hướng đề xuất phù hợp:** kiểm chứng lợi ích ảnh bổ sung bằng VLM frozen trước; chỉ học chính sách acquisition khi đã có tín hiệu utility đủ tin cậy. Pilot 20–30 cặp, vài câu hỏi/cặp dùng để kiểm tra pipeline và độ lớn hiệu ứng; chưa đủ để khẳng định khác biệt thống kê hay novelty cho bài báo.

## Các điểm cần cập nhật

1. **Ảnh đã được nối lại.** Workspace hiện có 297 ảnh VQA trong `vqa-dataset/image-cache`, cùng 310 ROI khớp segment doctor2. Việc còn thiếu là xác minh quan hệ cặp và khả năng trả lời từng câu hỏi, không còn là chưa có pixel ảnh. Link S3 trong workbook có hạn dùng; thí nghiệm dùng cache thay vì phụ thuộc signed URL.
2. **Cross-view hiện có 5 category, không phải 6.** Gồm C1.1, C1.2, C1.3, C2.1, C2.2. C1.4 xuất hiện trong QA đơn ảnh, phải dùng gold tọa độ riêng từng ảnh nếu phát triển grounding.
3. **1.794 episode tương ứng 897 cặp–category theo hai chiều.** Runner gộp hai chiều trước rồi chạy đủ conditions. Không chạy lại E1–E5 trên cả hai dòng đảo chiều và tính chúng như mẫu độc lập.
4. **Bổ sung BA đồng thời.** So sánh AB với BA giúp tách hiệu ứng thứ tự ảnh trong prompt khỏi hiệu ứng history ở A→B/B→A. Cả hai phép so sánh cần cùng câu hỏi, gold, seed và ngân sách ảnh/pixel.
5. **Xác nhận cặp chưa đủ xác nhận câu hỏi.** Cần duyệt vùng đích, tính answerable và reference theo category. Cặp cùng ca/cùng gold không suy ra cùng tổn thương hoặc cùng khảo sát. Đặc biệt, C2.2 không nên loại toàn bộ ca nghi ngờ/hậu phẫu một cách máy móc; cần xác định mục tiêu, nhãn quan sát được và báo rõ nhóm loại trừ. Chưa duyệt thì mặc định không đưa vào tập chính thức.
6. **History phải là free-running.** Lượt bổ sung giữ ảnh đầu tiên, câu hỏi ban đầu và câu trả lời thật của mô hình; không dùng gold làm câu trả lời đầu, không mô phỏng predicted history bằng template gold. Baseline A/B được tái sử dụng đúng cho history tương ứng.
7. **Chấm nhãn y khoa không chỉ là exact text.** Chuẩn hóa ontology, từ đồng nghĩa, multi-label, phủ định và câu không xác định. Exact-set matching hiện là baseline bảo thủ; phải báo parse failure. Với C2.2 cần adjudication, không kết luận mô hình sai chỉ vì khác từ.
8. **Ràng buộc causal và split trước khi học policy.** `delta` đo được là utility của một model/prompt/condition cụ thể, không phải gold STOP/REQUEST phổ quát. Policy không truy cập ảnh B, gold, nhãn audit hay kết quả A→B khi quyết định test. Không đưa classification label C1/C2 vào policy nếu ứng dụng thực tế không biết category đó; cần báo rõ giả định hoặc phân loại category từ câu hỏi.
9. **Policy cần baseline cùng ngân sách và validation threshold.** Never / Always / Random / Confidence, cộng ablation Question-only / Combined. Chọn threshold trên val; không dùng test để tạo utility labels huấn luyện hoặc chỉnh alias/prompt. Oracle theo utility chỉ là upper bound, không phải baseline triển khai được.
10. **Báo Rescue/Harm với mẫu số rõ ràng.** Rescue = sai→đúng, Harm = đúng→sai, Preserved = đúng→đúng, Unresolved = sai→sai. Có thể báo trên toàn bộ task và có điều kiện theo đúng/sai ban đầu; hai cách không được dùng lẫn. Sửa bảng trong góp ý đang mất các ký hiệu 0/1.
11. **Khoảng tin cậy theo bệnh nhân, phân tích paired.** Các cặp/câu hỏi/ảnh chung bệnh nhân và hai chiều cùng pair có phụ thuộc. Report cả số bệnh nhân, số cặp và số cặp–category. Nếu chỉ 20 cặp mà tập trung vào ít bệnh nhân thì CI đặc biệt yếu.
12. **Chi phí nhiều hơn số ảnh.** Sequential VLM xử lý lại history có thể tốn nhiều token/compute hơn AB dù đều mở 2 ảnh. Bản code hiện ghi thời gian từng generate và số ảnh; phân tích paper cần thêm vision/text tokens và đo thời gian GPU đồng bộ nếu so sánh latency.

## Mốc thực hiện

- Đã có: package, dynamic settings, notebook Colab, chuẩn bị dữ liệu, xác minh theo category, patient split, inference/resume, phân tích lexical utility và smoke tests.
- Tiếp theo: điền xác minh 20–30 cặp; chạy một model nhỏ để kiểm tra cấu hình ảnh/prompts trên Colab; cố định revision và số pixel; rồi chạy model thứ hai bằng adapter tương ứng.
- Sau pilot: sửa ontology/scorer qua train/val, đánh giá ảnh hưởng resolution và thứ tự, mở rộng tập xác minh. Chỉ khi có tín hiệu utility mới thực hiện policy learning.
- Chưa thực hiện: VLM GPU inference, clinical adjudication, policy training và khẳng định kết quả khoa học.

## Tài liệu kỹ thuật dùng cho triển khai

- [Colab FAQ](https://research.google.com/colaboratory/faq.html): GPU và giới hạn runtime có thể thay đổi; lưu checkpoint/kết quả ngoài runtime.
- [Qwen2.5-VL trong Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl): đa ảnh, chat template và cấu hình pixel budget.
- [Qwen2.5-VL-3B model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct): checkpoint baseline được cấu hình mặc định; không chứng minh năng lực chẩn đoán X-quang.
- [Bitsandbytes](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes): tùy chọn giảm memory; không đồng nghĩa quantization không ảnh hưởng kết quả.
