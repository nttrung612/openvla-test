# Thử OpenVLA 7B

Ví dụ dự đoán **một bước hành động** từ ảnh và chỉ dẫn tiếng Anh với hai checkpoint Hugging Face: [OpenVLA 7B](https://huggingface.co/openvla/openvla-7b) và [OpenVLA 7B fine-tuned LIBERO-Spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial). Script tải từng model riêng, không cần tải base trước khi dùng checkpoint fine-tuned.

## Cài trên server Linux với RTX 5090

Cần [uv](https://docs.astral.sh/uv/getting-started/installation/), driver NVIDIA hỗ trợ CUDA 12.8 và đủ dung lượng đĩa để lưu từng checkpoint. uv sẽ tìm hoặc tải Python 3.10 khi tạo môi trường. RTX 5090 cần PyTorch build CUDA 12.8 trở lên hỗ trợ `sm_120`; không dùng nguyên môi trường PyTorch 2.2 của OpenVLA trên A100. Ví dụ cài trong môi trường mới:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
uv pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128 && \
  uv pip install -r requirements.txt
python -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); print(torch.ones(1, device="cuda") + 1)'
```

Kiểm tra kết quả có GPU và `sm_120`. Môi trường này kết hợp PyTorch mới cho 5090 với Transformers phiên bản OpenVLA yêu cầu; cần kiểm chứng trên server vì khác môi trường A100 tác giả sử dụng. Không cần cài Flash Attention cho ví dụ này. Model BF16 có gần 8 tỉ tham số: chuẩn bị khoảng 16 GB VRAM cho trọng số và thêm bộ nhớ cho inference; dung lượng thực tế phụ thuộc môi trường.

### Khi tải thư viện CUDA bị timeout

Nếu tải `nvidia-cusolver-cu12==11.7.2.55` từ `pypi.nvidia.com` bị timeout dù đã tăng thời gian chờ, có thể cài trước đúng phiên bản đó từ [PyPI](https://pypi.org/project/nvidia-cusolver-cu12/11.7.2.55/), rồi chạy lại lệnh cài PyTorch trong môi trường đã kích hoạt:

```bash
uv pip install --no-deps nvidia-cusolver-cu12==11.7.2.55 --index-url https://pypi.org/simple && \
  uv pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128 && \
  uv pip install -r requirements.txt
```

`--no-deps` ở bước đầu chỉ tải riêng cuSOLVER; lệnh cài PyTorch tiếp theo sẽ bổ sung các dependency CUDA theo phiên bản PyTorch yêu cầu. Chỉ tiếp tục chạy model sau khi bước cài PyTorch thành công. Dòng `Resolved ... packages` chỉ cho biết đã giải dependency, chưa xác nhận cài thành công.

### Khi PyTorch báo driver quá cũ

Nếu log cài đặt hiển thị PyTorch khác `2.7.1+cu128` cùng CUDA 13 và lỗi driver, cài lại đúng build CUDA 12.8 trong môi trường hiện tại:

```bash
source .venv/bin/activate
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
uv pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128 && \
  uv pip install -r requirements.txt
uv pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
```

Kết quả cần có `2.7.1+cu128`, `12.8`, `True` và phép tính tensor trên GPU thành công. `requirements.txt` khóa cả phiên bản PyTorch và build CUDA để tránh cài nhầm bản mới từ PyPI nếu bước cài PyTorch trước đó thất bại. Nếu build đúng vẫn báo driver quá cũ, kiểm tra `nvidia-smi` trên server để xác định phiên bản driver.

## Lấy ảnh từ LIBERO-Spatial

Trong môi trường đã cài ở trên, cài LIBERO và các gói mô phỏng sau. Clone vào thư mục `LIBERO/` của repo; không tải bộ demonstration dataset.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv pip install -e ./LIBERO
uv pip install 'robosuite==1.4.1' 'mujoco==2.3.7' 'bddl==1.0.1' easydict cloudpickle 'gym==0.25.2' pyyaml
```

Trên server Linux không có màn hình, dùng EGL để render. Lần đầu LIBERO có thể hỏi đường dẫn dataset: nhập `n` để dùng mặc định. Script chọn task 0 và trạng thái đầu 0, đợi 10 bước cho cảnh ổn định, sau đó lưu ảnh camera và câu lệnh tương ứng:

```bash
export MUJOCO_GL=egl
python capture_libero.py --task-id 0 --state-id 0
```

Kết quả nằm ở `samples/libero/scene.png`, `instruction.txt` và `metadata.json`. Thử task hoặc trạng thái khác bằng `--task-id` và `--state-id`; dùng `--output-dir` để lưu riêng. Script xoay ảnh 180 độ theo cách chuẩn bị camera của OpenVLA và resize bằng PIL; kết quả không đảm bảo giống từng pixel với bước JPEG/TensorFlow của evaluation chính thức. File trạng thái ban đầu được tải từ checkout LIBERO bằng `torch.load(..., weights_only=False)`, vì vậy chỉ dùng checkout LIBERO đáng tin cậy.

## Chạy thử

Chuẩn bị một ảnh RGB của workspace robot, ví dụ `scene.png`:

```bash
python inference.py --model base --image scene.png --instruction "pick up the object"
python inference.py --model spatial --image scene.png --instruction "pick up the object" --center-crop
```

Hoặc dùng trực tiếp ảnh và câu lệnh vừa trích từ LIBERO:

```bash
python inference.py --model spatial --image samples/libero/scene.png --instruction-file samples/libero/instruction.txt --center-crop
python inference.py --model base --image samples/libero/scene.png --instruction-file samples/libero/instruction.txt
```

Base chưa fine-tune cho LIBERO: kết quả trên cảnh này chỉ là thử inference, không phải đánh giá công bằng hay bảo đảm robot thực hiện được task.

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
