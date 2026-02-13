# Nvidia Toolkit Ansible Project

Complete automation for Nvidia GPU infrastructure setup, supporting CUDA development, machine
learning training, and local model inference.

## Features

- **Nvidia Driver Installation**: Automated driver setup with nouveau conflict resolution
- **CUDA Toolkit**: Full CUDA development environment with cuDNN and NCCL
- **ML Frameworks**: PyTorch and TensorFlow with GPU acceleration
- **Inference Tools**: Ollama, llama.cpp, and optional vLLM for local model inference
- **Validation**: Automated testing to ensure all components work correctly

## Project Structure

```
nvidia-toolkit/
├── ansible.cfg             # Ansible configuration with optimizations
├── README.md               # This file
├── scaffold_project.sh     # Script to initialize role structure
├── playbooks/
│   ├── main.yml            # Master orchestration playbook
│   ├── drivers.yml         # Nvidia driver installation
│   ├── cuda-toolkit.yml    # CUDA toolkit setup
│   ├── ml-frameworks.yml   # PyTorch/TensorFlow installation
│   └── inference-tools.yml # Ollama/llama.cpp setup
├── roles/
│   ├── nvidia-driver/      # Kernel modules and driver management
│   ├── cuda/               # CUDA toolkit and libraries
│   ├── ml-python/          # Python ML environments (Conda/PyTorch/TF)
│   └── inference-stack/    # Local inference tools
├── inventory/
│   └── hosts.yml           # Target host definitions
└── group_vars/
    └── all.yml             # Global configuration variables
```

## Prerequisites

### Target System Requirements

1. **Hardware**
   - Nvidia GPU (compute capability 3.5 or higher recommended)
   - Minimum 8GB RAM (16GB+ recommended for ML training)
   - 50GB+ free disk space

2. **Operating System**:

   Currently supported (RedHat family):
   - Fedora 38 / 39 / 40+
   - RHEL 8 / 9
   - Rocky Linux 8 / 9
   - AlmaLinux 8 / 9

   > **Roadmap**: Support for Debian-based distributions (Ubuntu 20.04/22.04/24.04 LTS, Debian
   > 11/12) is planned for a future release.

3. **System Access**
   - SSH access with sudo privileges
   - Python 3.8+ installed
   - Internet connectivity for package downloads

### Control Machine Requirements

1. **Ansible**

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ansible

   # Fedora
   sudo dnf install ansible

   # Or via pip
   pip install ansible
   ```

   - Required version: Ansible Core 2.12+

2. **SSH Access**
   - SSH key-based authentication configured
   - User with sudo privileges on target hosts

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd nvidia-toolkit

# Make scaffold script executable
chmod +x scaffold_project.sh

# Run scaffold to initialize role structure
./scaffold_project.sh
```

### 2. Configure Inventory

Edit `inventory/hosts.yml` to define your target hosts:

```yaml
all:
  children:
    gpu_nodes:
      hosts:
        localhost:
          ansible_connection: local

        # Or remote hosts:
        # gpu-server-01:
        #   ansible_host: 192.168.1.100
        #   ansible_user: ubuntu
```

### 3. Customize Variables

Edit `group_vars/all.yml` to configure versions and features:

```yaml
# Key variables to review:
nvidia_driver_version: "latest"
cuda_version: "13.1"
install_inference: true
install_monitoring: true
```

### 4. Run the Playbook

```bash
# Full installation
ansible-playbook playbooks/main.yml

# Test connectivity first
ansible all -m ping

# Run specific components only
ansible-playbook playbooks/main.yml --tags driver,cuda
```

## Usage Examples

### Running Specific Components

```bash
# Install only Nvidia driver
ansible-playbook playbooks/main.yml --tags driver

# Install CUDA toolkit
ansible-playbook playbooks/main.yml --tags cuda

# Setup ML frameworks
ansible-playbook playbooks/main.yml --tags ml

# Setup inference tools
ansible-playbook playbooks/main.yml --tags inference

# Run validation tests only
ansible-playbook playbooks/main.yml --tags validation
```

### Local Installation

For installing on the current machine:

```bash
# Use localhost in inventory
ansible-playbook -i "localhost," -c local playbooks/main.yml
```

### Skip Specific Roles

```bash
# Skip ML frameworks
ansible-playbook playbooks/main.yml --skip-tags ml

# Skip inference tools
ansible-playbook playbooks/main.yml --skip-tags inference
```

## Configuration Guide

### Common Customizations

#### 1. CUDA Version Selection

Edit `group_vars/all.yml`:

```yaml
cuda_version: "13.1" # or "11.8", "12.2", etc.
```

**Note on Fedora Support**: NVIDIA's CUDA repositories often trail the latest Fedora release. This
automation includes a fallback mechanism: if your Fedora version is newer than the available NVIDIA
repositories (currently Fedora 42), it will automatically use the repository for the latest
supported version. For example, Fedora 43 systems will use the Fedora 42 CUDA repository. You can
check/modify the `cuda_fedora_max_version` variable in `roles/cuda/defaults/main.yml` when NVIDIA
releases updated repositories.

## Post-Installation Validation

### Manual Tests

```bash
# 1. Check GPU detection
nvidia-smi

# 2. Verify CUDA compiler
nvcc --version

# 3. Test CUDA sample (if installed)
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# 4. Test PyTorch GPU access
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 5. Test TensorFlow GPU access
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 6. Test Ollama (if installed)
ollama list
ollama run llama2:7b "Hello, world!"
```

### Using Conda Environments

```bash
# Activate PyTorch environment
conda activate pytorch
python -c "import torch; print(torch.cuda.is_available())"

# Activate TensorFlow environment
conda activate tensorflow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Troubleshooting

### Common Issues

#### 1. Nouveau Driver Conflict

**Symptoms**: Driver installation fails, `nvidia-smi` not found

**Solution**: The playbook automatically blacklists nouveau. If issues persist:

```bash
# Verify nouveau is blacklisted
cat /etc/modprobe.d/blacklist-nouveau.conf

# Rebuild initramfs
sudo update-initramfs -u  # Ubuntu/Debian
sudo dracut --force        # Fedora/RHEL

# Reboot
sudo reboot
```

#### 2. Secure Boot Enabled

**Symptoms**: Nvidia driver fails to load

**Solution**:

- Disable Secure Boot in BIOS/UEFI, or
- Sign the Nvidia kernel modules with your MOK key

#### 3. CUDA Version Mismatch

**Symptoms**: PyTorch/TensorFlow can't find CUDA libraries

**Solution**: Ensure CUDA version matches framework requirements:

```bash
# Check installed CUDA
nvcc --version

# Verify environment variables
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

#### 4. Kernel Update Breaks Driver

**Symptoms**: After kernel update, nvidia-smi fails

**Solution**: Reinstall driver or use DKMS:

```bash
# Check DKMS status
dkms status

# Rebuild for new kernel
sudo dkms autoinstall
```

## Development and Contributing

### Adding Custom Roles

1. Create role structure:

```bash
ansible-galaxy role init my-custom-role --init-path roles
```

2. Add to playbook:

```yaml
roles:
  - role: my-custom-role
    tags: ["custom"]
```

### Testing Changes

```bash
# Syntax check
ansible-playbook playbooks/main.yml --syntax-check

# Check mode (dry-run)
ansible-playbook playbooks/main.yml --check

# Limit to specific hosts
ansible-playbook playbooks/main.yml --limit gpu-server-01
```

## Advanced Topics

### Multi-GPU Configuration

For systems with multiple GPUs, the setup automatically detects all available GPUs. Control GPU
visibility:

```bash
# Set specific GPUs for a process
CUDA_VISIBLE_DEVICES=0,1 python train.py

# Verify GPU topology
nvidia-smi topo -m
```

### Model Management

```bash
# HuggingFace cache location
export HF_HOME=/opt/models/huggingface

# Download models
huggingface-cli download meta-llama/Llama-2-7b-hf

# Ollama models
ollama pull llama2:7b
ollama list
```

### Monitoring GPU Usage

```bash
# Real-time monitoring
watch nvidia-smi

# Detailed GPU info
nvidia-smi -q

# Monitor specific GPU
nvidia-smi -i 0
```

## Security Considerations

1. **SSH Access**: Use key-based authentication only
2. **Firewall**: Restrict access to Jupyter (port 8888) and other services
3. **Jupyter**: Always set a password: `jupyter notebook password`
4. **Secure Boot**: Consider implications before disabling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Documentation: [Wiki](https://github.com/your-repo/wiki)

## Acknowledgments

- Nvidia CUDA Toolkit
- Ansible Community
- PyTorch and TensorFlow teams
