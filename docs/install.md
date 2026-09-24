# Download and installation

To get started with NAVSIM:

### 1. Clone the navsim-devkit

Clone the repository

```bash
git clone https://github.com/autonomousvision/navsim.git
cd navsim
```

### 2. Download the dataset

You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.
You can download these splits with the following scripts.

```bash
./download_mini
./download_trainval
./download_test
./download_warmup_two_stage
./download_navhard_two_stage
./download_private_test_hard_two_stage
```

Also, the script `./download_navtrain` can be used to download a small portion of the  `trainval` dataset split which is needed for the `navtrain` training split.

This will download the splits into the download directory. From there, move it to create the following structure.

```angular2html
~/navsim_workspace
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         └── private_test_hard.pkl
    │    └── mini
    └── sensor_blobs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         ├──  CAM_B0
    |    |         ├──  CAM_F0
    |    |         ├──   ...
    |    └── mini
    └── navhard_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── warmup_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── private_test_hard_two_stage
         ├── openscene_meta_datas
         └── sensor_blobs

```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

⏰ **Note:** The `navhard_two_stage` split is used for local testing of your model's performance in a two-stage pseudo closed-loop setup.
In contrast, `warmup_two_stage` is a smaller dataset designed for validating and testing submissions to the [Hugging Face Warmup leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup).
In other words, the results you obtain locally on `warmup_two_stage` should match the results you see after submitting to Hugging Face.
`private_test_hard_two_stage` contains the challenge data.
You will need it to generate a `submission.pkl` in order to participate in the official challenge on the [Hugging Face CPVR 2025 leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-internal) (for more details, see [Submission](submission.md)).

### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

### NAVSIM-v2 GPU evaluation on H20 (Python 3.12)

The `environment.yml` above keeps the original Python 3.9 setup. For the H20 GPU
evaluation, create a separate environment from the root of this repository:

```bash
conda create -n nav-v2 -c conda-forge python=3.12.13 pip -y
conda activate nav-v2
python -m pip install --upgrade pip
python -m pip install 'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' \
   --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

The requirements select Python 3.12-compatible versions automatically. Install
PyTorch first: the CUDA 12.8 wheels are hosted on the PyTorch index, not the
default package index. Avoid using `conda env create -f environment.yml` for
`nav-v2`, since that file installs Python 3.9.

Check the interpreter, CUDA kernel execution, and local evaluation imports:

```bash
python - <<'PY'
import sys
import torch
import torchvision
from navsim.planning.script.gpu_inference import predict_trajectories
from navsim.planning.script.run_pdm_score import run_pdm_score

assert sys.version_info[:3] == (3, 12, 13)
assert torch.__version__ == '2.10.0+cu128'
assert torch.version.cuda == '12.8'
assert torchvision.__version__ == '0.25.0+cu128'
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), (torch.ones(2, device='cuda:0') + 1).cpu().tolist())
PY
```

The GPU evaluation launcher uses `nav-v2` by default; the CPU launcher keeps its
original Python 3.9 interpreter. Once the dataset and metric cache are ready,
run the complete GPU evaluation from the repository root with:

```bash
scripts/evaluation/run_drivor_nav2_gpu_nohup.sh
```

The launcher predicts on four visible GPUs (`cuda:0` through `cuda:3`), with a
batch size of four per GPU. The resulting trajectories are scored by the existing
Ray CPU workers. To run on one GPU instead, add `gpu_num_devices=1` to the command.
`gpu_device` selects the first visible device, and `gpu_num_devices` selects how
many consecutive devices to use; `CUDA_VISIBLE_DEVICES` can remap GPU indices.

Set `PYTHON_BIN` to override the GPU interpreter when the environment is
installed in a different location.
