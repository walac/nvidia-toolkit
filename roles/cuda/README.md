# cuda

Installs NVIDIA CUDA Toolkit and deep learning libraries from NVIDIA's official repository.

## What It Does

- Adds NVIDIA CUDA repository with driver packages excluded (uses RPM Fusion drivers)
- Installs CUDA toolkit, cuDNN, and NCCL libraries
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
| `cuda_version` | `12.3` | CUDA version to install |
| `cuda_fedora_max_version` | `42` | Latest Fedora version with NVIDIA repo support |
| `cuda_home` | `/usr/local/cuda-{version}` | CUDA installation path |
| `cuda_symlink` | `/usr/local/cuda` | Symlink to CUDA installation |
| `cuda_create_symlink` | `true` | Create /usr/local/cuda symlink |
| `cuda_install_samples` | `true` | Install CUDA samples |

### Default Packages

```yaml
cuda_packages:
  - cuda-toolkit-{version}
  - libcudnn8
  - libcudnn8-devel
  - libnccl
  - libnccl-devel
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
