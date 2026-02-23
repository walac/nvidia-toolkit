# inference-stack

Installs Ollama for local LLM inference with GPU acceleration.

## What It Does

- Queries the GitHub API for the latest Ollama release
- Compares against the installed version (if any)
- Installs or upgrades Ollama using the official install script

## Requirements

- Fedora 40+ (RedHat family systems)
- nvidia-driver and cuda roles should be applied first for GPU support
- Internet access to download Ollama

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
