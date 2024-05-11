"""Install requirements for WD14-tagger."""
import os
import sys
import subprocess
from launch import run  # pylint: disable=import-error

from launch import run, python

try:
    # NVIDIAのシステム管理インターフェイスコマンドを実行してCUDAバージョンを取得
    output = subprocess.check_output("nvidia-smi", encoding='utf-8')
    # 出力からCUDAバージョンの行を探し、バージョン番号を抽出
    version_line = [line for line in output.split('\n') if "CUDA Version" in line][0]
    cuda_version = version_line.split(':')[1].strip()
    return cuda_version
except Exception as e:
    print(f"Error obtaining CUDA version: {e}", file=sys.stderr)
    return None

if cuda_version.startswith("11."):
        cudnn_package = "nvidia-cudnn-cu11==9.1.0.70"
    elif cuda_version.startswith("12."):
        cudnn_package = "nvidia-cudnn-cu12==9.1.0.70"
    else:
        print(f"No compatible cuDNN package for CUDA version {cuda_version}", file=sys.stderr)
        return

# cuDNNパッケージをインストール
subprocess.run([sys.executable, "-m", "pip", "install", cudnn_package])

# まずONNXと関連ライブラリをインストール
run(f'"{python}" -m pip uninstall onnxruntime -y', live=True)
run(f'"{python}" -m pip uninstall onnxruntime-gpu -y', live=True)
subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime", "onnx-graphsurgeon"], check=True)
# CUDAバージョンを取得
cuda_version = get_cuda_version()
if cuda_version:
    # 取得したCUDAバージョンに基づいてcuDNNをインストール
    install_cudnn(cuda_version)
else:
    print("Failed to detect CUDA version, cannot install cuDNN", file=sys.stderr)

run(f'"{python}" -m pip install onnx onnxruntime onnx-graphsurgeon', live=True)

NAME = "WD14-tagger"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")
