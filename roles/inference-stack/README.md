# inference-stack

Installs Ollama for local LLM inference with GPU acceleration.

## What It Does

- Downloads and installs Ollama binary
- Creates dedicated ollama system user and group
- Sets up data and model directories
- Configures systemd service with security hardening

## Requirements

- Fedora 40+ (RedHat family systems)
- nvidia-driver and cuda roles should be applied first for GPU support
- Internet access to download Ollama

## Role Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ollama_version` | `0.16.1` | Ollama version to install |
| `ollama_arch` | auto-detected | Architecture for download (`amd64` or `arm64`) |
| `ollama_install_dir` | `/opt/ollama` | Installation directory for extracted files |
| `ollama_user` | `ollama` | System user for the service |
| `ollama_group` | `ollama` | System group for the service |
| `ollama_data_dir` | `/var/lib/ollama` | Data storage directory |
| `ollama_models_dir` | `/var/lib/ollama/models` | Model storage directory |

## Dependencies

- nvidia-driver (recommended)
- cuda (recommended)

## Example Playbook

```yaml
- hosts: gpu_servers
  become: true
  roles:
    - nvidia-driver
    - cuda
    - inference-stack
```

## Usage

After installation, pull and run models:

```bash
# Check service status
systemctl status ollama

# Pull a model
ollama pull llama2:7b

# List models
ollama list

# Run inference
ollama run llama2:7b "Hello, world!"

# API endpoint
curl http://localhost:11434
```

## Troubleshooting

```bash
# View service logs
journalctl -u ollama -f

# Check GPU detection
ollama run llama2:7b --verbose 2>&1 | grep -i gpu
```

## License

MIT
