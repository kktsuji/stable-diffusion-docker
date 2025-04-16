# Stable Diffusion with Docker

## Usage

Build the Docker image on WSL.

```bash
docker build -t kktsuji/stable-diffusion-cuda12.8.0 .
```

Run the container with GPU support to execute the python script.

```bash
# Execute the python script on the container
docker run --rm -it --gpus all -v $PWD:/work -w /work kktsuji/stable-diffusion-cuda12.8.0 python3 main.py
```
