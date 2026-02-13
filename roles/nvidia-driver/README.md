# nvidia-driver

Installs Nvidia proprietary drivers from RPM Fusion on Fedora systems.

## What It Does

- Enables RPM Fusion Free and Non-Free repositories
- Installs akmod-nvidia for automatic kernel module building
- Configures nvidia-persistenced service

## Requirements

- Fedora 40+ (RedHat family systems)
- Nvidia GPU detected by the system
- Internet access to download packages

## Role Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `nvidia_driver_packages` | See defaults | List of packages to install |

### Default Packages

```yaml
nvidia_driver_packages:
  - akmod-nvidia
  - xorg-x11-drv-nvidia-cuda
  - xorg-x11-drv-nvidia-power
  - nvidia-settings
  - libvdpau
```

## Dependencies

None.

## Example Playbook

```yaml
- hosts: gpu_servers
  become: true
  roles:
    - nvidia-driver
```

## Notes

- A reboot is typically required after driver installation
- The role uses akmods which automatically rebuilds kernel modules on kernel updates
- Secure Boot users must enroll the MOK key for signed modules

## License

MIT
