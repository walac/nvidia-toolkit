# Nvidia Toolkit Ansible Project

Complete automation for Nvidia GPU infrastructure setup, supporting CUDA development and local
model inference.

## Features

- **Nvidia Driver Installation**: Automated driver setup via RPM Fusion
- **CUDA Toolkit**: Full CUDA development environment
- **Inference Tools**: Ollama for local model inference

## Supported Systems

Currently supported:

- Fedora 40+

## Quick Start

```bash
ansible-playbook -K playbooks/main.yml
```

## Available Tags

Run specific components using tags:

```bash
ansible-playbook playbooks/main.yml --tags driver      # Nvidia driver only
ansible-playbook playbooks/main.yml --tags cuda        # CUDA toolkit
ansible-playbook playbooks/main.yml --tags inference   # Inference tools
```

### Fedora CUDA Repository Fallback

NVIDIA's CUDA repositories often trail the latest Fedora release. This automation includes a
fallback mechanism: if your Fedora version is newer than the available NVIDIA repositories
(currently Fedora 42), it will automatically use the repository for the latest supported version.

You can check/modify the `cuda_fedora_max_version` variable in `roles/cuda/defaults/main.yml`
when NVIDIA releases updated repositories.

## Post-Installation Validation

### Nvidia Driver

Verify the driver is loaded and the GPU is detected:

```bash
nvidia-smi
```

Expected output shows GPU name, driver version, and memory usage. If this fails, check that
nouveau is blacklisted and reboot the system.

### CUDA Toolkit

Verify the CUDA compiler is available:

```bash
nvcc --version
```

Test CUDA functionality by compiling and running a sample program:

```bash
cat > /tmp/cuda_test.cu << 'EOF'
#include <stdio.h>
__global__ void hello() { printf("Hello from GPU!\n"); }
int main() { hello<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
EOF
nvcc /tmp/cuda_test.cu -o /tmp/cuda_test && /tmp/cuda_test
```

### Inference Tools

```bash
# Test Ollama (if installed)
ollama list
```

### Verifying ML Framework GPU Access

After manually installing PyTorch or TensorFlow:

```bash
# Test PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Test TensorFlow GPU access
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
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

RPM Fusion packages automatically handle nouveau blacklisting via dracut. If nouveau
is still loaded after driver installation, reboot the system.

### Secure Boot

If the Nvidia driver fails to load, either disable Secure Boot in BIOS/UEFI or sign the Nvidia
kernel modules with your MOK key.

## License

MIT License - see [LICENSE](LICENSE) for details.
