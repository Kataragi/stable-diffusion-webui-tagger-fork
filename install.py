"""Install requirements for WD14-tagger."""
import os
import sys
import subprocess
from launch import run  # pylint: disable=import-error

from launch import run, python

# ONNX Runtime GPUバージョンをアンインストール
subprocess.run([python, '-m', 'pip', 'uninstall', 'onnxruntime-gpu', '-y'])
# 最新バージョンをインストール
subprocess.run(['python', '-m', 'pip', 'install', 'onnxruntime-gpu'])

# ONNX Runtimeバージョンをアンインストール
subprocess.run([python, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'])
# 最新バージョンをインストール
subprocess.run(['python', '-m', 'pip', 'install', 'onnxruntime'])

# 必要なONNX関連ライブラリをインストール
subprocess.run([python, '-m', 'pip', 'install', 'onnx', 'onnxruntime', 'onnx-graphsurgeon'])

# 特定のCUDAバージョン用のcuDNNをインストール
subprocess.run([python, '-m', 'pip', 'install', 'nvidia-cudnn-cu11==9.1.0.70', 'nvidia-cudnn-cu12==9.1.0.70'])

# 特定のCUDAバージョン用のCUDAランタイムをインストール
subprocess.run([python, '-m', 'pip', 'install', 'nvidia-cuda-runtime-cu11', 'nvidia-cuda-runtime-cu12'])

NAME = "WD14-tagger"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")
