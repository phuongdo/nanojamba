# nanoJamba

Blog: https://medium.com/@phuongdoviet/arc-agi-with-nano-jamba-2227152a96b6


#### Setup Local Conda

```bash
git clone https://github.com/phuongdo/nanojamba.git
cd nanojamba
conda create -y --name nanojamba python=3.11
conda activate nanojamba
pip install -r requirements.txt
git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install -e . && cd .
```


### Download Dataset

```bash
cd nanojamba
git clone https://github.com/fchollet/ARC-AGI.git
```


#### Training

```bash
python3 train.py

```

####  References
- https://github.com/karpathy/build-nanogpt
- https://github.com/state-spaces/mamba
- https://github.com/fchollet/ARC-AGI