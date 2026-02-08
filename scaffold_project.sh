#!/bin/bash
# scaffold_project.sh
# Automates Phase 1: Project Structure Setup for Nvidia Toolkit

set -e

echo "Starting Nvidia Toolkit Project Scaffolding..."

# 1. Create Base Directories
echo "Creating directory structure..."
mkdir -p playbooks
mkdir -p inventory
mkdir -p group_vars
mkdir -p roles

# 2. Initialize Roles using ansible-galaxy
# We use --init-path to ensure they land in the roles/ directory
ROLES=("nvidia-driver" "cuda" "container-runtime" "ml-python" "inference-stack")

for role in "${ROLES[@]}"; do
    if [ ! -d "roles/$role" ]; then
        echo "Initializing role: $role"
        ansible-galaxy role init "$role" --init-path roles > /dev/null
    else
        echo "Role $role already exists, skipping."
    fi
done

# 3. Create Placeholder Playbooks
PLAYBOOKS=("drivers.yml" "cuda-toolkit.yml" "containers.yml" "ml-frameworks.yml" "inference-tools.yml")

echo "Creating playbook placeholders..."
for pb in "${PLAYBOOKS[@]}"; do
    if [ ! -f "playbooks/$pb" ]; then
        echo "---" > "playbooks/$pb"
        echo "- name: Configure ${pb%.*}" >> "playbooks/$pb"
        echo "  hosts: all" >> "playbooks/$pb"
        echo "  become: true" >> "playbooks/$pb"
        echo "  roles:" >> "playbooks/$pb"
        # Map playbook to role broadly for now
        case $pb in
            drivers.yml) echo "    - nvidia-driver" >> "playbooks/$pb" ;;
            cuda-toolkit.yml) echo "    - cuda" >> "playbooks/$pb" ;;
            containers.yml) echo "    - container-runtime" >> "playbooks/$pb" ;;
            ml-frameworks.yml) echo "    - ml-python" >> "playbooks/$pb" ;;
            inference-tools.yml) echo "    - inference-stack" >> "playbooks/$pb" ;;
        esac
    fi
done

echo "Scaffolding complete!"
echo ""
echo "Next steps:"
echo "  1. Review and customize group_vars/all.yml"
echo "  2. Update inventory/hosts.yml with your target hosts"
echo "  3. Start implementing role tasks in roles/*/tasks/main.yml"
