# cuda

Installs NVIDIA CUDA Toolkit from NVIDIA's official repository.

## What It Does

- Adds NVIDIA CUDA repository with driver packages excluded (uses RPM Fusion drivers)
- Installs CUDA toolkit
- Creates `/usr/local/cuda` symlink
- Sets up environment variables in `/etc/profile.d/cuda.sh`
- Handles Fedora version fallback when NVIDIA repos lag behind

## Requirements

- Fedora 40+ (RedHat family systems)
- nvidia-driver role must be applied first
- Internet access to download packages

## Role Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `cuda_fedora_max_version` | `42` | Latest Fedora version with NVIDIA repo support |
| `cuda_repo_arch` | auto-detected | Repository architecture (`x86_64` or `sbsa` for ARM) |
| `cuda_repo_gpgkey_id` | `D42D0685` | NVIDIA CUDA repository GPG key identifier |
| `cuda_home` | `/usr/local/cuda-{version}` | CUDA installation path |
| `cuda_symlink` | `/usr/local/cuda` | Symlink to CUDA installation |

### Default Packages

```yaml
cuda_packages:
  - cuda-toolkit-{version}
```

Note: cuDNN and NCCL are not available in NVIDIA's CUDA repository for Fedora.
See the section below for installation instructions.

## Installing cuDNN

cuDNN (CUDA Deep Neural Network library) provides GPU-accelerated primitives for
deep learning frameworks like PyTorch and TensorFlow. Install it via conda:

```bash
# Create a conda environment with cuDNN
conda create -n ml python=3.11
conda activate ml

# Install cuDNN from NVIDIA channel
conda install -c nvidia cudnn

# Or install with PyTorch (includes cuDNN automatically)
# Replace the pytorch-cuda version to match your installed cuda_version
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
```

Verify installation:

```bash
python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
```

## Fedora Version Fallback

NVIDIA's CUDA repositories often lag behind Fedora releases. If your Fedora version
is newer than `cuda_fedora_max_version`, the role automatically uses the repository
for the latest supported version.

Update `cuda_fedora_max_version` when NVIDIA releases repos for newer Fedora versions.

## Dependencies

- nvidia-driver (must be installed first)

## Example Playbook

```yaml
- hosts: gpu_servers
  become: true
  roles:
    - nvidia-driver
    - cuda
```

## Verification

After installation:

```bash
nvcc --version
```

## License

MIT
