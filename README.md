
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

Build Human3.6M random cross-sequence pairs and convert the real 3DPW
train/validation pairs:

```
python -m dataset.data_prepocess ^
  --datasets-root D:\datasets ^
  --datasets h36m 3dpw mupots ^
  --target-fps 30 ^
  --seed 42
```

Human3.6M is decoded from its standard 99-D exponential-map representation
by integrating the root trajectory and applying forward kinematics, resampled
from 50 Hz to 30 Hz, and paired with a different randomly shuffled clip. These
artificial pairs retain `synthetic=True` and the legacy
`interaction_valid=0` provenance flag. The Stage-1 loader no longer treats that
flag as a supervision mask: every enabled H-H source trains the bidirectional
cross-attention interaction token, the history/future intent KL, the future
token decoder, and intent-conditioned forecasting. CMU interaction and 3DPW
remain recorded synchronous pairs and receive the configurable higher sampling
weight (`intent.real_pair_sampling_weight`) so synthetic combinations broaden
motion coverage without overwhelming real interaction semantics. Source
balancing is enabled by default (`intent.balance_sources: true`), so a much
larger AMASS folder cannot starve the smaller CMU and 3DPW interaction sources.
Existing AMASS/Human3.6M NPZ files do not need to be regenerated; the source
configuration now makes their trajectory pairs intent-training eligible.

The interaction token is pooled from bidirectional spatial-temporal
cross-attention outputs. It is joint-count agnostic, so the Stage-1
human-human token encoder is transferred directly to Stage-2 human-robot
fine-tuning even when the robot has a different number of joints. The intent
latent is a conditional variational latent: the prior uses the historical
interaction token, the posterior uses historical and future interaction
tokens, and their distributions are matched by KL divergence.

Stage-2 loads these token/latent modules from the Stage-1 checkpoint and
fine-tunes them with a smaller learning rate
(`intent.pretrained_semantic_lr_scale: 0.1`) while the robot-specific branch
adapts at the normal Stage-2 rate. This reduces catastrophic forgetting without
freezing the H-H interaction representation completely.

Because the interaction-token definition now contains trainable
spatial-temporal cross-attention, Stage-1 checkpoints created before this
change must be retrained before they can be used for Stage-2.

Raw AMASS stores SMPL-H rotations rather than 3-D joints. Download the
licensed **Extended SMPL+H model** used by AMASS and extract it as
`body_models/smplh/{female,male,neutral}/model.npz`. Joint-only recovery is
implemented directly in NumPy, so it does not require `smplx`, PyTorch, or
generation of the 6,890 mesh vertices:

```
python -m dataset.data_prepocess ^
  --datasets-root D:\datasets ^
  --datasets amass ^
  --smplh-model-dir D:\datasets\amass\body_models
```

MuPoTS-3D is retained as a test-only multi-person pose benchmark and is not
used to train the human-intent latent.

You can train the model as follows:

Stage-1 H-H pretraining:

```
python pretrain.py --cfg config/h2h_pretrain_cfg.yml ^
  --work-dir ckpt_h2h_pretrain
```

Stage-2 H-R fine-tuning with the new Stage-1 checkpoint:

```
python train.py --stage 2 ^
  --model-pth ckpt_h2h_pretrain/pretrain_stage1_final.pth ^
  --seed 888 --exp-name HARPER_result.txt ^
  --layer-norm-axis spatial --with-normalization
```
where config files are located at `config/harper_config.yml`.
