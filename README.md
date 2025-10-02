# Visualize Block Logic — Version 1 for ML

![Python](https://img.shields.io/badge/Python-%3E%3D3.8-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![License](https://img.shields.io/badge/License-Apache--2.0-orange)

> Công cụ trực quan hóa logic khối cho mô hình Machine Learning. Tập trung vào Blocks view (chọn khối, xem/sửa code). Chức năng 3D tạm vô hiệu hóa để tối ưu trải nghiệm và hiệu năng.

---

## Tính năng chính
- Trực quan hóa pipeline ML dạng khối (Blocks) rõ ràng, gọn đẹp.
- Chọn khối bằng click để xem chi tiết; không kéo–thả để tránh nhầm lẫn.
- Tích hợp chỉnh sửa code nhanh: kết nối giữa khối và `ml_visualizer.py`/`ml_code_editor.py`.
- Kiến trúc đơn giản, dễ mở rộng: phù hợp học tập, demo, hoặc làm nền tảng cho dự án lớn.

> Lộ trình: bật lại 3D, auto-layout, zoom/pan canvas, phím tắt di chuyển khối, và xuất cấu hình mô hình.

---

## Demo nhanh
- Khởi chạy ứng dụng:

```bash
cd app_visualize_ML
python -u ml_code_editor.py
```

- Giao diện mặc định: "🧩 Live Block Visualization".
- Các nút/khung 3D đã được vô hiệu hóa tạm thời.

---

## Yêu cầu hệ thống
- Python 3.8+ (khuyến nghị 3.10+)
- Môi trường chạy tiêu chuẩn (không cần phụ thuộc nặng). Nếu có lỗi hiển thị, vui lòng cập nhật Python và Tk.

---

## Cấu trúc thư mục
```
app_visualize_ML/
├── ml_code_editor.py   # Khởi chạy UI, toolbar, và logic chuyển view
└── ml_visualizer.py    # Vẽ và xử lý tương tác Blocks view
```

> Toàn bộ mã nguồn tập trung ở `app_visualize_ML/`. Các phần 3D đã được "stub" để tiện bật lại sau này.

---

## Sử dụng
- Mở ứng dụng, chọn khối trong Blocks view để xem/điều chỉnh logic.
- Tránh kéo–thả: hành vi này đã tắt theo thiết kế hiện tại.
- Nếu muốn bật lại 3D trong tương lai: khôi phục các hàm `init_3d_view`, `switch_to_3d_view`, và render 3D.

---

## Lộ trình phát triển
- [ ] Bật lại 3D với điều khiển mượt (rotate/zoom/pan)
- [ ] Auto-layout các khối theo graph
- [ ] Zoom/Pan canvas, minimap
- [ ] Phím tắt di chuyển/nhóm khối
- [ ] Xuất cấu hình mô hình (JSON/YAML)
- [ ] Unit tests cơ bản và CI

---

## Đóng góp
Chúng tôi hoan nghênh mọi đóng góp!

1. Fork repository và tạo nhánh theo chuẩn:
   - `feat/<ten-tinh-nang>` hoặc `fix/<mo-ta-ngan>`
2. Giữ code style hiện có; thay đổi tối thiểu và có mục tiêu rõ ràng.
3. Gửi Pull Request kèm mô tả súc tích; nếu có thể, đính kèm hình ảnh/gif UI.
4. Trao đổi trong Issues khi cần thảo luận trước.

> Đóng góp thuộc phạm vi license của dự án. Vui lòng tôn trọng các tệp và cấu trúc hiện tại.

---

## License
Dự án phát hành theo **Apache License 2.0**. Xem tệp `LICENSE` trong repository để biết chi tiết.

---

## Ghi nhận
- Cảm ơn cộng đồng ML/DS đã truyền cảm hứng.
- Dự án hướng tới trải nghiệm học tập, demo nhanh, và mở rộng thực dụng.