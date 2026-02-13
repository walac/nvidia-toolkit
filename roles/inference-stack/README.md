# inference-stack

Installs Ollama for local LLM inference with GPU acceleration.

## What It Does

- Downloads and installs Ollama binary
- Creates dedicated ollama system user and group
- Sets up data and model directories
- Configures systemd service for automatic startup
- Optionally pre-pulls specified models

## Requirements

- Fedora 40+ (RedHat family systems)
- nvidia-driver and cuda roles should be applied first for GPU support
- Internet access to download Ollama and models

## Role Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ollama_version` | `0.16.1` | Ollama version to install |
| `ollama_install_dir` | `/opt/ollama` | Installation directory for extracted files |
| `ollama_user` | `ollama` | System user for the service |
| `ollama_group` | `ollama` | System group for the service |
| `ollama_data_dir` | `/var/lib/ollama` | Data storage directory |
| `ollama_models_dir` | `/var/lib/ollama/models` | Model storage directory |
| `ollama_host` | `127.0.0.1` | Listen address |
| `ollama_port` | `11434` | Listen port |
| `ollama_prepull_models` | `[]` | Models to download during install |

### Pre-pulling Models

To automatically download models during installation:

```yaml
ollama_prepull_models:
  - llama2:7b
  - codellama:7b
```

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

After installation:

```bash
# Check service status
systemctl status ollama

# List models
ollama list

# Pull a model
ollama pull llama2:7b

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
