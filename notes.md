## 环境安装
```bash
# CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.2.0
pip install -r requirements.txt
pip install -v -e .

pip install numpy==1.26.4
pip install modelscope==1.15.0
```

添加了： `palm_recongnition\palm_roi_ext\2ROI.py` 提取 ROI 区域并保存

模型路径 `alm_roi_ext\hand_key_points\model` 
