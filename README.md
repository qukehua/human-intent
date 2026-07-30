
<h1 align="center">  </h1>





##  Environment
The project is developed under the following environment:
- Python 3.10 
- PyTorch 2.1
- CUDA 12.1




For installation of the project dependencies, please run:
```
conda create -n human_to_robot python=3.10
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
``` 


##  Training

Convert the paired CMU human-interaction ASF/AMC files into the Stage-1
`person_a/person_b` NPZ format:

```
python -m dataset.cmu_interaction_converter ^
  --input-root D:\datasets\cmu_mocap\cmu_mocap\human_interaction ^
  --output-dir D:\datasets\cmu_mocap\cmu_mocap\human_interaction\data_aug ^
  --target-fps 30
```

The converter performs ASF/AMC forward kinematics, converts CMU lengths and
root translations to metres, applies anti-aliased 120 Hz to 30 Hz
downsampling, and retains both people in the shared capture coordinate frame.

You can train the model as follows:
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name HARPER_result.txt --layer-norm-axis spatial --with-normalization
```
where config files are located at `config/harper_config.yml`.
