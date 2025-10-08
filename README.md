# Visualize Block Logic — Version 1 for ML

> Công cụ trực quan hoá pipeline ML theo dạng khối (Blocks) và lớp (Layers), giúp học và dạy Machine Learning nhanh – vui – dễ nhớ.

---

## Điểm nhấn
- Trực quan hoá quy trình xử lý ảnh digit MNIST từ `Input → Convolution → ReLU → Pool → Flatten → Dense → Softmax`.
- Tab `🧠 Layers` với canvas vẽ 28×28, nút `Predict` và `Clear`, thanh điều chỉnh tốc độ animation.
- Đồ thị 2×3 hiển thị: `Input`, `Conv`, `ReLU`, `Pool`, `Flatten` và `Output Prob (0–9)`.
- Animation kernel 3×3 quét ảnh (cyan bounding box), sau đó hé lộ dần các thanh xác suất 10 lớp.
- Tích hợp phím tắt (ví dụ `F7`) và toolbar để thao tác nhanh.
- Kiến trúc Python đơn giản, dễ mở rộng, phù hợp cho lớp học và workshop.

---

## Demo nhanh
1) Mở ứng dụng: `python ml_code_editor.py`
2) Vào tab `🧠 Layers` → vẽ con số bạn muốn thử.
3) Bấm `Predict` hoặc nhấn `F7` để chạy animation.
4) Quan sát từng giai đoạn hiển thị và kết quả xác suất cho 10 lớp.

---

## Tính năng chi tiết
- `Canvas vẽ 28×28`: mô phỏng ảnh đầu vào, tương thích pipeline MNIST.
- `Conv → ReLU → Pool`: hiển thị ảnh tại mỗi bước xử lý; khung cyan minh hoạ kernel trượt.
- `Flatten`: biến ma trận sau Pool thành vector, hiển thị thành dải pixel.
- `Dense + Softmax`: tính logits (giả lập) và hiển thị xác suất 10 lớp bằng các thanh bar; animation hé lộ dần.
- `Tốc độ animation`: điều chỉnh mượt hơn cho việc trình diễn.
- `Clear`: xoá canvas để thử lại nhanh.

> Lưu ý: Phần Dense/Softmax hiện giả lập bằng tham số khởi tạo cố định nhằm phục vụ minh hoạ trực quan. Có thể thay thế bằng mô hình thật (PyTorch/TensorFlow) trong roadmap.

---

## Cách chạy
- Yêu cầu: Python 3.9+ và `matplotlib`.
- Chạy tại thư mục `app_visualize_ML/`:

```bash
python ml_code_editor.py
```

- Nếu gặp lỗi font/hiển thị trên Windows, hãy cập nhật `matplotlib` và driver đồ hoạ.

---

## Phím tắt & Toolbar
- `F7`: chạy nhanh dự đoán/animation trong tab `Layers`.
- `🧠 Layers`: mở tab Layers từ toolbar.
- `▶ Predict`, `🧹 Clear`: nút thao tác trực tiếp trong tab.

---

## Kiến trúc tối giản
- `ml_code_editor.py`: UI chính, xử lý sự kiện, khởi tạo tab `Layers`, điều khiển animation.
- `ml_visualizer.py`: tiện ích vẽ/visualize, có thể tái sử dụng.
- Thiết kế chia nhỏ state: `layers_input`, `layers_conv`, `layers_relu`, `layers_pool`, `layers_flat`, `layers_logits`, `layers_probs`… giúp kiểm soát và mở rộng dễ dàng.

---

## Mở rộng đề xuất (Roadmap)
- Tuỳ chọn kernel/stride/padding và hiển thị heatmap.
- Kết nối mô hình thật (PyTorch/TensorFlow) để thay tham số giả lập.
- Hiệu ứng chuyển cảnh (camera/flow pipes) giữa các panel để thành một “bộ phim” pipeline.
- Lưu/ghi lại video quá trình học để chia sẻ.

---

## Đóng góp
Mọi ý tưởng/PR đều được hoan nghênh. Hãy mở issue kèm mô tả ngắn gọn: mục tiêu, ảnh hưởng UI, và test case cần thiết.

---

## Bản quyền
Apache-2.0 (xem `LICENSE`).