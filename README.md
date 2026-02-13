# Nvidia Toolkit Ansible Project

Complete automation for Nvidia GPU infrastructure setup, supporting CUDA development, machine
learning training, and local model inference.

## Features

- **Nvidia Driver Installation**: Automated driver setup with nouveau conflict resolution
- **CUDA Toolkit**: Full CUDA development environment with cuDNN and NCCL
- **ML Frameworks**: PyTorch and TensorFlow with GPU acceleration
- **Inference Tools**: Ollama, llama.cpp, and optional vLLM for local model inference
- **Validation**: Automated testing to ensure all components work correctly

## Supported Systems

Currently supported (RedHat family):
- Fedora 38 / 39 / 40+
- RHEL 8 / 9
- Rocky Linux 8 / 9
- AlmaLinux 8 / 9

> **Roadmap**: Support for Debian-based distributions (Ubuntu 20.04/22.04/24.04 LTS, Debian
> 11/12) is planned for a future release.

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd nvidia-toolkit

# Configure your target hosts in inventory/hosts.yml
# Adjust variables in group_vars/all.yml

# Run the full installation
ansible-playbook playbooks/main.yml
```

## Available Tags

Run specific components using tags:

```bash
ansible-playbook playbooks/main.yml --tags driver      # Nvidia driver only
ansible-playbook playbooks/main.yml --tags cuda        # CUDA toolkit
ansible-playbook playbooks/main.yml --tags ml          # ML frameworks
ansible-playbook playbooks/main.yml --tags inference   # Inference tools
ansible-playbook playbooks/main.yml --tags validation  # Run validation tests
```

## Configuration

Key variables in `group_vars/all.yml`:

```yaml
nvidia_driver_version: "latest"
cuda_version: "13.1"
install_inference: true
install_monitoring: true
```

### Fedora CUDA Repository Fallback

NVIDIA's CUDA repositories often trail the latest Fedora release. This automation includes a
fallback mechanism: if your Fedora version is newer than the available NVIDIA repositories
(currently Fedora 42), it will automatically use the repository for the latest supported version.

You can check/modify the `cuda_fedora_max_version` variable in `roles/cuda/defaults/main.yml`
when NVIDIA releases updated repositories.

## Post-Installation Validation

```bash
# Check GPU detection
nvidia-smi

# Verify CUDA compiler
nvcc --version

# Test PyTorch GPU access
conda activate pytorch
python -c "import torch; print(torch.cuda.is_available())"

# Test TensorFlow GPU access
conda activate tensorflow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Test Ollama (if installed)
ollama list
```

## Monitoring GPU Usage

```bash
# Real-time monitoring
watch nvidia-smi

# Detailed GPU info
nvidia-smi -q

# Monitor specific GPU
nvidia-smi -i 0
```

## Troubleshooting

### Nouveau Driver Conflict

The playbook automatically blacklists nouveau. If issues persist after installation:

```bash
# Verify nouveau is blacklisted
cat /etc/modprobe.d/blacklist-nouveau.conf

# Rebuild initramfs and reboot
sudo dracut --force
sudo reboot
```

### Secure Boot

If the Nvidia driver fails to load, either disable Secure Boot in BIOS/UEFI or sign the Nvidia
kernel modules with your MOK key.

### Kernel Update Breaks Driver

After a kernel update, rebuild the driver modules:

```bash
dkms status
sudo dkms autoinstall
```

## License

MIT License - see [LICENSE](LICENSE) for details.
