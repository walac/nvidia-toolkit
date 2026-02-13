# monitoring

Installs GPU monitoring tools for observing GPU utilization and performance.

## What It Does

- Installs nvtop (interactive GPU monitor, like htop for GPUs)
- Installs btop (modern system monitor with GPU support)
- Optionally deploys a full Prometheus/Grafana monitoring stack

## Requirements

- Fedora 40+ (RedHat family systems)
- nvidia-driver role should be applied first

## Role Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `monitoring_cli_tools` | `[nvtop, btop]` | CLI tools to install |
| `install_monitoring_stack` | `false` | Deploy Prometheus/Grafana stack |
| `prometheus_port` | `9090` | Prometheus listen port |
| `grafana_port` | `3000` | Grafana listen port |

## Dependencies

- nvidia-driver (recommended)

## Example Playbook

```yaml
- hosts: gpu_servers
  become: true
  roles:
    - nvidia-driver
    - monitoring
```

## Usage

After installation:

```bash
# Interactive GPU monitoring (like htop for GPUs)
nvtop

# Modern system monitor with GPU support
btop

# Nvidia's built-in monitoring
nvidia-smi

# Continuous monitoring (1 second interval)
nvidia-smi -l 1
```

## Optional Monitoring Stack

Enable full monitoring with Prometheus and Grafana:

```yaml
install_monitoring_stack: true
```

This deploys to `~/gpu-monitoring/` with:
- DCGM Exporter (Nvidia GPU metrics)
- Prometheus (metrics collection)
- Grafana (visualization)

Start the stack:

```bash
cd ~/gpu-monitoring
podman-compose up -d
```

Access:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## License

MIT
