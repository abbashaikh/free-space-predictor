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
From the package's root directory run
```sh
pip install -e .
```

<!-- CUDA Toolkit with pytorch -->

