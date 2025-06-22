# Stable Diffusion with Docker

## Prerequisites

1. Install NVIDIA GPU driver (see [NVIDIA Driver Downloads](https://www.nvidia.com/en-us/drivers/))
2. Install WSL2 to Windows (see [my post](https://tsuji.tech/install-uninstall-wsl/))
3. Install Docker to WSL (see [my post](https://tsuji.tech/install-docker-to-wsl/))
4. Install NVIDIA Container Toolkit (see [my post](https://tsuji.tech/use-nvidia-gpu-with-wsl-docker/))

## Usage

Build the Docker image on WSL.

```bash
docker build -t kktsuji/stable-diffusion-cuda12.8.0 .
```

Create a python virtual environment and install the required packages.

```bash
docker run --rm -it --gpus all -v $PWD:/work -w /work kktsuji/stable-diffusion-cuda12.8.0 bash ./setup-python-env.sh
```

Execute the python script on the container with GPU.

```bash
docker run --rm -it --gpus all -v $PWD:/work -w /work kktsuji/stable-diffusion-cuda12.8.0 ./venv/bin/python train_lora.py
```
