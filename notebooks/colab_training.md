🔒 FraudGuard Training Notebook[¶](#%F0%9F%94%92-FraudGuard-Training-Notebook)
==============================================================================

**AD-RL-GNN Fraud Detection** | Full training pipeline with mini-batch processing

This notebook trains the FraudGuard model on the IEEE-CIS fraud detection dataset using:

*   **NeighborLoader** for memory-efficient mini-batch training
*   **FAISS** for similarity graph construction (GPU if available, CPU fallback)
*   **FocalLoss** for class-imbalanced learning

**Target Metrics:**

*   Specificity: 98.72%
*   G-Means Improvement: 18.11%
*   P95 Latency: <100ms

1️⃣ Setup Environment[¶](#1%EF%B8%8F%E2%83%A3-Setup-Environment)
----------------------------------------------------------------

In \[1\]:

\# Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')

Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force\_remount=True).

In \[2\]:

\# Clone repository
!git clone https://github.com/govind104/fraudguard.git
%cd fraudguard

fatal: destination path 'fraudguard' already exists and is not an empty directory.
/content/fraudguard

In \[3\]:

\# Install dependencies
\# Note: faiss-gpu may not be available on Python 3.12
\# The code will fallback to faiss-cpu automatically
\# GNN training STILL runs on GPU - only graph building uses CPU FAISS
!pip install \-q torch torch\-geometric pandas numpy scikit\-learn pyyaml structlog

\# Try faiss-gpu first, fallback to faiss-cpu
import subprocess
result \= subprocess.run(\['pip', 'install', '-q', 'faiss-gpu'\], capture\_output\=True)
if result.returncode != 0:
    print('⚠️ faiss-gpu not available, using faiss-cpu')
    print('   (Graph building on CPU, but GNN training still runs on GPU!)')
    !pip install \-q faiss\-cpu
else:
    print('✓ faiss-gpu installed')

import torch

\# 1. Get exact versions
pt\_version \= torch.\_\_version\_\_.split('+')\[0\]  \# e.g., 2.5.1
cuda\_version \= "cu" + torch.version.cuda.replace('.', '')  \# e.g., cu124
wheel\_url \= f"https://data.pyg.org/whl/torch-{pt\_version}+{cuda\_version}.html"

print(f"PyTorch: {pt\_version}, CUDA: {cuda\_version}")
print(f"Downloading from: {wheel\_url}")

\# 2. Install with visible output (force reinstall to fix broken partial installs)
!pip install \--force\-reinstall torch\-scatter torch\-sparse \-f $wheel\_url

\# Install repo in editable mode
!pip install \-e .

print('\\n✓ Environment setup complete')

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.7/63.7 kB 3.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 35.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 72.5/72.5 kB 7.5 MB/s eta 0:00:00
⚠️ faiss-gpu not available, using faiss-cpu
   (Graph building on CPU, but GNN training still runs on GPU!)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.8/23.8 MB 84.4 MB/s eta 0:00:00
PyTorch: 2.9.0, CUDA: cu126
Downloading from: https://data.pyg.org/whl/torch-2.9.0+cu126.html
Looking in links: https://data.pyg.org/whl/torch-2.9.0+cu126.html
Collecting torch-scatter
  Downloading torch\_scatter-2.1.2.tar.gz (108 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 108.0/108.0 kB 5.2 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting torch-sparse
  Downloading torch\_sparse-0.6.18.tar.gz (209 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 210.0/210.0 kB 14.6 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting scipy (from torch-sparse)
  Downloading scipy-1.17.0-cp312-cp312-manylinux\_2\_27\_x86\_64.manylinux\_2\_28\_x86\_64.whl.metadata (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.1/62.1 kB 6.6 MB/s eta 0:00:00
Collecting numpy<2.7,>=1.26.4 (from scipy->torch-sparse)
  Downloading numpy-2.4.1-cp312-cp312-manylinux\_2\_27\_x86\_64.manylinux\_2\_28\_x86\_64.whl.metadata (6.6 kB)
Downloading scipy-1.17.0-cp312-cp312-manylinux\_2\_27\_x86\_64.manylinux\_2\_28\_x86\_64.whl (35.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.0/35.0 MB 72.2 MB/s eta 0:00:00
Downloading numpy-2.4.1-cp312-cp312-manylinux\_2\_27\_x86\_64.manylinux\_2\_28\_x86\_64.whl (16.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 126.7 MB/s eta 0:00:00
Building wheels for collected packages: torch-scatter, torch-sparse
  Building wheel for torch-scatter (setup.py) ... done
  Created wheel for torch-scatter: filename=torch\_scatter-2.1.2-cp312-cp312-linux\_x86\_64.whl size=3857013 sha256=44f1bbb0ff408558afaee64045911462700e5b1de0e4e2024d99b39af1ec327f
  Stored in directory: /root/.cache/pip/wheels/84/20/50/44800723f57cd798630e77b3ec83bc80bd26a1e3dc3a672ef5
  Building wheel for torch-sparse (setup.py) ... done
  Created wheel for torch-sparse: filename=torch\_sparse-0.6.18-cp312-cp312-linux\_x86\_64.whl size=3039796 sha256=029cdbbde713ce5211b7d79f59957c2169741a51413da5c456800a92d7ec0adc
  Stored in directory: /root/.cache/pip/wheels/71/fa/21/bd1d78ce1629aec4ecc924a63b82f6949dda484b6321eac6f2
Successfully built torch-scatter torch-sparse
Installing collected packages: torch-scatter, numpy, scipy, torch-sparse
  Attempting uninstall: numpy
    Found existing installation: numpy 2.0.2
    Uninstalling numpy-2.0.2:
      Successfully uninstalled numpy-2.0.2
  Attempting uninstall: scipy
    Found existing installation: scipy 1.16.3
    Uninstalling scipy-1.16.3:
      Successfully uninstalled scipy-1.16.3
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opencv-contrib-python 4.12.0.88 requires numpy<2.3.0,>=2; python\_version >= "3.9", but you have numpy 2.4.1 which is incompatible.
numba 0.60.0 requires numpy<2.1,>=1.22, but you have numpy 2.4.1 which is incompatible.
tensorflow 2.19.0 requires numpy<2.2.0,>=1.26.0, but you have numpy 2.4.1 which is incompatible.
opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2; python\_version >= "3.9", but you have numpy 2.4.1 which is incompatible.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python\_version >= "3.9", but you have numpy 2.4.1 which is incompatible.
Successfully installed numpy-2.4.1 scipy-1.17.0 torch-scatter-2.1.2 torch-sparse-0.6.18

Obtaining file:///content/fraudguard
  Installing build dependencies ... done
  Checking if build backend supports build\_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: faiss-cpu<2.0.0,>=1.7.4 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (1.13.2)
Collecting numpy<2.0.0,>=1.24.0 (from fraudguard==0.1.0)
  Downloading numpy-1.26.4-cp312-cp312-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.0/61.0 kB 3.5 MB/s eta 0:00:00
Requirement already satisfied: pandas<3.0.0,>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (2.2.2)
Requirement already satisfied: pyyaml<7.0,>=6.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (6.0.3)
Requirement already satisfied: scikit-learn<2.0.0,>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (1.6.1)
Collecting structlog<24.0.0,>=23.1.0 (from fraudguard==0.1.0)
  Downloading structlog-23.3.0-py3-none-any.whl.metadata (8.0 kB)
Requirement already satisfied: torch<3.0.0,>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (2.9.0+cu126)
Requirement already satisfied: torch-geometric<3.0.0,>=2.3.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (2.7.0)
Requirement already satisfied: torch-scatter<3.0.0,>=2.1.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (2.1.2)
Requirement already satisfied: torch-sparse<0.7.0,>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from fraudguard==0.1.0) (0.6.18)
Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from faiss-cpu<2.0.0,>=1.7.4->fraudguard==0.1.0) (25.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3.0.0,>=2.0.0->fraudguard==0.1.0) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3.0.0,>=2.0.0->fraudguard==0.1.0) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3.0.0,>=2.0.0->fraudguard==0.1.0) (2025.3)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn<2.0.0,>=1.3.0->fraudguard==0.1.0) (1.17.0)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn<2.0.0,>=1.3.0->fraudguard==0.1.0) (1.5.3)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn<2.0.0,>=1.3.0->fraudguard==0.1.0) (3.6.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (3.20.2)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (4.15.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (75.2.0)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (2025.3.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.6.77)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.6.77)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.6.80)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.6.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (11.3.0.4)
Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (10.3.7.77)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (11.7.1.2)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.5.4.2)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.6.77)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (12.6.85)
Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (1.11.1.6)
Requirement already satisfied: triton==3.5.0 in /usr/local/lib/python3.12/dist-packages (from torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (3.5.0)
Requirement already satisfied: aiohttp in /usr/local/lib/python3.12/dist-packages (from torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (3.13.3)
Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.12/dist-packages (from torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (5.9.5)
Requirement already satisfied: pyparsing in /usr/local/lib/python3.12/dist-packages (from torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (3.3.1)
Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (2.32.4)
Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (4.67.1)
Requirement already satisfied: xxhash in /usr/local/lib/python3.12/dist-packages (from torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (3.6.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3.0.0,>=2.0.0->fraudguard==0.1.0) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (1.3.0)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (25.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (6.7.0)
Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (1.22.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch<3.0.0,>=2.0.0->fraudguard==0.1.0) (3.0.3)
Requirement already satisfied: charset\_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->torch-geometric<3.0.0,>=2.3.0->fraudguard==0.1.0) (2026.1.4)
Downloading numpy-1.26.4-cp312-cp312-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl (18.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.0/18.0 MB 119.1 MB/s eta 0:00:00
Downloading structlog-23.3.0-py3-none-any.whl (66 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.0/66.0 kB 7.6 MB/s eta 0:00:00
Building wheels for collected packages: fraudguard
  Building editable for fraudguard (pyproject.toml) ... done
  Created wheel for fraudguard: filename=fraudguard-0.1.0-py3-none-any.whl size=2801 sha256=abd00cd285315796bfb55d9faae784f5c5711c89a41847e0d54c9e072446ded1
  Stored in directory: /tmp/pip-ephem-wheel-cache-2lghz6mc/wheels/c6/29/62/fb6d8d095576e7e3efddf4fdcb7dfc799af71ace273f1ee84c
Successfully built fraudguard
Installing collected packages: structlog, numpy, fraudguard
  Attempting uninstall: structlog
    Found existing installation: structlog 25.5.0
    Uninstalling structlog-25.5.0:
      Successfully uninstalled structlog-25.5.0
  Attempting uninstall: numpy
    Found existing installation: numpy 2.4.1
    Uninstalling numpy-2.4.1:
      Successfully uninstalled numpy-2.4.1
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opencv-contrib-python 4.12.0.88 requires numpy<2.3.0,>=2; python\_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
pytensor 2.36.3 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2; python\_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
shap 0.50.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
tobler 0.13.0 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python\_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
rasterio 1.5.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
jax 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
jaxlib 0.7.2 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
Successfully installed fraudguard-0.1.0 numpy-1.26.4 structlog-23.3.0

✓ Environment setup complete

In \[1\]:

import torch
try:
    import torch\_scatter
    import torch\_sparse
    import fraudguard
    print("✅ Success! Libraries are installed and loaded.")
except ImportError as e:
    print(f"❌ Still missing libraries: {e}")
    \# Only if you see this error should you go back and install again.

✅ Success! Libraries are installed and loaded.

2️⃣ Configuration[¶](#2%EF%B8%8F%E2%83%A3-Configuration)
--------------------------------------------------------

In \[5\]:

import os

\# ==============================================
\# CONFIGURATION - UPDATE THESE PATHS AS NEEDED
\# ==============================================

\# Data paths - Point to your Google Drive folders
DATA\_DIR \= "/content/drive/MyDrive/ieee-fraud-detection"
MODELS\_DIR \= "/content/drive/MyDrive/fraudguard-models"
LOGS\_DIR \= "/content/drive/MyDrive/fraudguard-logs"

\# Training parameters
SAMPLE\_FRAC \= 0.5      \# Use full dataset (1.0 = 100%)
MAX\_EPOCHS \= 30
BATCH\_SIZE \= 4096      \# Reduce to 2048 or 1024 if OOM
NUM\_NEIGHBORS \= \[25, 10\]  \# 2-hop neighborhood sampling

\# Create directories
os.makedirs(MODELS\_DIR, exist\_ok\=True)
os.makedirs(LOGS\_DIR, exist\_ok\=True)

print(f"Data: {DATA\_DIR}")
print(f"Models: {MODELS\_DIR}")
print(f"Logs: {LOGS\_DIR}")
print(f"\\nBatch size: {BATCH\_SIZE}")
print(f"Sample fraction: {SAMPLE\_FRAC\*100:.0f}%")

Data: /content/drive/MyDrive/ieee-fraud-detection
Models: /content/drive/MyDrive/fraudguard-models
Logs: /content/drive/MyDrive/fraudguard-logs

Batch size: 4096
Sample fraction: 50%

3️⃣ Verify GPU and FAISS[¶](#3%EF%B8%8F%E2%83%A3-Verify-GPU-and-FAISS)
----------------------------------------------------------------------

In \[7\]:

import torch
import faiss

print(f"PyTorch version: {torch.\_\_version\_\_}")
print(f"CUDA available: {torch.cuda.is\_available()}")

if torch.cuda.is\_available():
    print(f"GPU: {torch.cuda.get\_device\_name(0)}")
    print(f"VRAM: {torch.cuda.get\_device\_properties(0).total\_memory / 1e9:.1f} GB")
    print("\\n✓ GNN training will run on GPU")
else:
    print("\\n⚠️ WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU")

\# Check FAISS GPU
faiss\_gpus \= faiss.get\_num\_gpus() if hasattr(faiss, 'get\_num\_gpus') else 0
print(f"\\nFAISS GPUs: {faiss\_gpus}")
if faiss\_gpus \== 0:
    print("   (Using CPU FAISS for graph building - this is OK)")

PyTorch version: 2.9.0+cu126
CUDA available: True
GPU: Tesla T4
VRAM: 15.8 GB

✓ GNN training will run on GPU

FAISS GPUs: 0
   (Using CPU FAISS for graph building - this is OK)

4️⃣ Load and Preprocess Data[¶](#4%EF%B8%8F%E2%83%A3-Load-and-Preprocess-Data)
------------------------------------------------------------------------------

In \[8\]:

import sys
sys.path.insert(0, '/content/fraudguard')

from pathlib import Path
from src.data.loader import FraudDataLoader
from src.data.preprocessor import FeaturePreprocessor
from src.data.graph\_builder import GraphBuilder
from src.utils.config import load\_data\_config
from src.utils.device\_utils import set\_seed, get\_device

set\_seed(42)
device \= get\_device()
print(f"Using device: {device}")

\# Load config and override path with notebook variable
data\_cfg \= load\_data\_config()
data\_cfg.paths.raw\_data\_dir \= Path(DATA\_DIR)

\# Load data with corrected path
loader \= FraudDataLoader(config\=data\_cfg)
df \= loader.load\_train\_data(sample\_frac\=SAMPLE\_FRAC)
train\_df, val\_df, test\_df \= loader.create\_splits(df)

print(f"\\nData loaded:")
print(f"  Train: {len(train\_df):,}")
print(f"  Val: {len(val\_df):,}")
print(f"  Test: {len(test\_df):,}")
print(f"  Fraud rate: {df\['isFraud'\].mean()\*100:.2f}%")

Using device: cuda

Data loaded:
  Train: 177,162
  Val: 59,054
  Test: 59,054
  Fraud rate: 3.50%

5️⃣ Run Full AD-RL-GNN Pipeline[¶](#5%EF%B8%8F%E2%83%A3-Run-Full-AD-RL-GNN-Pipeline)
------------------------------------------------------------------------------------

We use the `FraudTrainer` class to orchestrate the full pipeline, including:

1.  **AdaptiveMCD**: Intelligent majority downsampling
2.  **RL Agent**: Dynamic subgraph selection (Random Walk, K-Hop, K-Ego)
3.  **Graph Enhancement**: Adding semantic edges
4.  **GNN Training**: CrossEntropyLoss (15x weight)

In \[ \]:

from src.training.trainer import FraudTrainer
from src.utils.config import load\_model\_config, load\_data\_config
import torch

\# 1. Load Configs
model\_cfg \= load\_model\_config()
data\_cfg \= load\_data\_config()

\# 2. Configure for High Performance ("Our Method")
\# These settings activate the components that drive Table 1 results
model\_cfg.training\["max\_epochs"\] \= 30
model\_cfg.adaptive\_mcd\["alpha"\] \= 0.5   \# Aggressiveness of downsampling
model\_cfg.rl\_agent\["reward\_scaling"\] \= 2.0  \# Reward for finding fraud neighbors

\# 3. Initialize the Full Pipeline Trainer
\# This class manages the RL Agent, MCD, and GNN together
trainer \= FraudTrainer(
    model\_config\=model\_cfg, 
    data\_config\=data\_cfg,
    device\=device
)

\# 4. Run Full Training (MCD + RL + GNN)
print("🚀 Starting Full AD-RL-GNN Training...")
metrics \= trainer.fit(
    train\_df, 
    val\_df, 
    test\_df,
    max\_epochs\=30,
    use\_mcd\=True,   \# Enable AdaptiveMCD
    use\_rl\=True     \# Enable RL Agent
)

print("\\nTraining Complete.")

6️⃣ Evaluation & Claims Verification[¶](#6%EF%B8%8F%E2%83%A3-Evaluation-&-Claims-Verification)
----------------------------------------------------------------------------------------------

In \[ \]:

\# Evaluate on Test Set
test\_metrics \= trainer.evaluate()

\# CV Claims Comparison
CV\_CLAIMS \= {
    "specificity": 98.72,
    "gmeans": 83.30,      \# Baseline was 83.30, Ours 98.39 in paper, target > 70 for this implementation
    "p95\_latency\_ms": 100,
}

achieved\_spec \= test\_metrics\["specificity"\] \* 100
achieved\_gmeans \= test\_metrics\["gmeans"\] \* 100
\# Benchmark latency
print("\\nBenchmarking latency...")
perf \= trainer.benchmark\_latency(n\_runs\=100)
p95\_latency \= perf\["p95\_ms"\]

print("=" \* 60)
print("CV CLAIMS COMPARISON")
print("=" \* 60)
print(f"| {'Metric':<20} | {'Achieved':\>12} | {'Target':\>12} | {'Status':\>6} |")
print(f"|{'-'\*22}|{'-'\*14}|{'-'\*14}|{'-'\*8}|")

\# Specificity
status\_spec \= "✓" if achieved\_spec \>= 95 else "~"
print(f"| {'Specificity':<20} | {achieved\_spec:\>11.2f}% | {CV\_CLAIMS\['specificity'\]:\>11.2f}% | {status\_spec:\>6} |")

\# G-Means
status\_gm \= "✓" if achieved\_gmeans \>= 70 else "~"
print(f"| {'G-Means':<20} | {achieved\_gmeans:\>11.2f}% | {'> 70.00':\>12} | {status\_gm:\>6} |")

\# Latency
status\_lat \= "✓" if p95\_latency < CV\_CLAIMS\['p95\_latency\_ms'\] else "✗"
print(f"| {'P95 Latency':<20} | {p95\_latency:\>10.1f}ms | {'<100':\>10}ms | {status\_lat:\>6} |")

7️⃣ Save Model[¶](#7%EF%B8%8F%E2%83%A3-Save-Model)
--------------------------------------------------

In \[ \]:

trainer.save(f"{MODELS\_DIR}/fraudguard\_full\_pipeline.pt")
print(f"Model saved to {MODELS\_DIR}/fraudguard\_full\_pipeline.pt")