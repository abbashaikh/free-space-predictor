# Free Space Social Navigation
Robot social navigation with probabilistic collision avoidance leveraging non-convex scenario optimization; augmented with a free space support predictor

## Installation ##
Clone the repository
```sh
git clone <repository cloning URL>
```

### Environment Setup ###
Create a conda environment and install dependencies
```sh
conda create --name fsp python=3.9 -y
conda activate fsp
pip install -r requirements.txt

# See note in requirements.txt for why we do this.
pip install --no-dependencies l5kit==1.5.0
```

### Local Packages ###
From the project's root directory run
```sh
pip install -e .
```

## Model Training ##
This repository makes use of [Weights & Biases](https://wandb.ai) for logging training information. Before running any of the following commands, please edit Lines 114 and 115 of `train.py` to specify your desired W&B project and entity names.

### Pedestrian Dataset ###
To train a model on the ETH and UCY Pedestrian datasets, execute the relevant command in `train_model.sh`

**NOTE:** Make sure that you specify the correct directory for `--trajdata_cache_dir` (where trajdata's cache is located). The provided values in the shell script are examples, but you can choose whatever suits your computing environment best.

For example, running this command from the project's root directory will train the trajectory prediction model on the ETH Univ dataset.

```sh
# To train a new model
torchrun --nproc_per_node=1 --master_port=29500 ./train.py \
        --eval_every=1 \
        --vis_every=1 \
        --batch_size=256 \
        --eval_batch_size=256 \
        --preprocess_workers=16 \
        --log_dir=./data/trained_models/trajectory_prediction \
        --train_epochs=1 \
        --conf=./config/pedestrians.json \
        --trajdata_cache_dir=./data/pedestrian_datasets/.unified_data_cache \
        --history_sec=2.8 --prediction_sec=4.8 \
        --K=1 \
        --alpha_init=1.54e-05 \
        --augment_input_noise=0.57 \
        --grad_clip=0.909 \
        --learning_rate=0.016 \
        --sigma_eps_init=0.0002 \
        --contrastive_weight=50.0 \
        --train_data=eupeds_eth-train \
        --eval_data=eupeds_eth-val \
        --log_tag=eth
```

The model can be trained on different datasets (or even multiple pedestrian datasets), first by modifying the `--train_data` and `--eval-data` arguments in the provided shell scripts, followed by updating the `data_dirs` on Line 171 of `train.py`. To choose the correct names for train and eval datasets refer to this table provided as part of [trajdata: A Unified Interface to Multiple Human Trajectory Datasets](https://github.com/NVlabs/trajdata/tree/main):

| Dataset | ID | Splits | Locations | Description | dt | Maps |
|---------|----|--------|------------|-------------|----|------|
| ETH - Univ | `eupeds_eth` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `zurich` | The ETH (University) scene from the ETH BIWI Walking Pedestrians dataset | 0.4s (2.5Hz) | |
| ETH - Hotel | `eupeds_hotel` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `zurich` | The Hotel scene from the ETH BIWI Walking Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Univ | `eupeds_univ` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The University scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Zara1 | `eupeds_zara1` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The Zara1 scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Zara2 | `eupeds_zara2` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The Zara2 scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |

