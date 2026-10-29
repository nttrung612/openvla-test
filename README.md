# Thử OpenVLA 7B

Ví dụ dự đoán **một bước hành động** từ ảnh và chỉ dẫn tiếng Anh với hai checkpoint Hugging Face: [OpenVLA 7B](https://huggingface.co/openvla/openvla-7b) và [OpenVLA 7B fine-tuned LIBERO-Spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial). Script tải từng model riêng, không cần tải base trước khi dùng checkpoint fine-tuned.

## Cài trên server Linux với RTX 5090

Cần Python 3.10, driver NVIDIA hỗ trợ CUDA 12.8 và đủ dung lượng đĩa để lưu từng checkpoint. RTX 5090 cần PyTorch build CUDA 12.8 trở lên hỗ trợ `sm_120`; không dùng nguyên môi trường PyTorch 2.2 của OpenVLA trên A100. Ví dụ cài trong môi trường mới:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())'
```

Kiểm tra kết quả có GPU và `sm_120`. Môi trường này kết hợp PyTorch mới cho 5090 với Transformers phiên bản OpenVLA yêu cầu; cần kiểm chứng trên server vì khác môi trường A100 tác giả sử dụng. Không cần cài Flash Attention cho ví dụ này. Model BF16 có gần 8 tỉ tham số: chuẩn bị khoảng 16 GB VRAM cho trọng số và thêm bộ nhớ cho inference; dung lượng thực tế phụ thuộc môi trường.

## Chạy thử

Chuẩn bị một ảnh RGB của workspace robot, ví dụ `scene.png`:

```bash
python inference.py --model base --image scene.png --instruction "pick up the object"
python inference.py --model spatial --image scene.png --instruction "pick up the object" --center-crop
```

Lần đầu chạy sẽ tải trọng số và mã model từ Hugging Face vào cache; `trust_remote_code=True` cho phép chạy mã do OpenVLA cung cấp. Kết quả JSON có vector hành động 7 phần tử `(x, y, z, roll, pitch, yaw, gripper)` đã được khôi phục theo thống kê dataset tương ứng: `bridge_orig` cho base và `libero_spatial` cho checkpoint fine-tuned. Có thể ghi đè bằng `--unnorm-key` nếu checkpoint có key phù hợp. Hai vector dùng thống kê khác nhau nên không nên so sánh trực tiếp như cùng hệ tọa độ hoặc thang đo.

`--center-crop` mô phỏng crop trung tâm 90% diện tích mà tác giả dùng cho LIBERO, nhưng dùng PIL nên không đảm bảo trùng từng pixel với pipeline TensorFlow chính thức. Chạy trên ảnh bất kỳ chỉ kiểm tra inference; không đo tỉ lệ thành công của robot hay LIBERO.

## Đánh giá trong LIBERO

Để chạy rollout thay vì một bước dự đoán, làm theo phần [LIBERO Setup của OpenVLA](https://github.com/openvla/openvla#libero-simulation-benchmark-evaluations), rồi trong repo OpenVLA chạy:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
```

Lệnh này chạy 1 episode cho mỗi task để kiểm tra nhanh; bỏ tùy chọn cuối để dùng mặc định 50 episode/task. Pipeline chính thức yêu cầu LIBERO và các dependencies bổ sung, bao gồm Flash Attention; khả năng tương thích của chúng với RTX 5090 cần kiểm tra riêng. Kết quả paper dùng A100 và các phiên bản cũ, vì vậy không mặc định tương đương trên 5090.
