#!/bin/bash
# SPDX-License-Identifier: MIT-0
#
# Invoked via ExecStartPost= on akmods@.service after a successful
# build. akmods@.service is triggered asynchronously (--no-block) by
# RPM Fusion's own 95-akmodsposttrans.install kernel-install plugin
# whenever a new kernel is installed, so this always runs outside of
# any dnf/rpm transaction.
#
# Rebuilds the initramfs for the given kernel only if NVIDIA modules
# exist for it, so kernels without the NVIDIA driver are left alone.
#
# Assumes a BLS/GRUB layout, where "dracut -f --kver" alone locates
# and regenerates the right standalone initramfs file. Unverified on
# UKI layouts, where the initramfs is embedded in a signed PE binary
# instead of a standalone file.
#
# A dracut failure is reported but not allowed to fail this script:
# ExecStartPost= failures mark akmods@.service itself as failed, which
# would misreport an initramfs refresh problem as a module build
# problem.

KERNEL_VERSION="$1"

[ -n "$KERNEL_VERSION" ] || exit 0

if find "/lib/modules/$KERNEL_VERSION" -name 'nvidia.ko*' -print -quit 2>/dev/null | grep -q .; then
    if ! dracut -f --kver "$KERNEL_VERSION" --add-drivers "nvidia nvidia_modeset nvidia_uvm nvidia_drm"; then
        echo "warning: dracut failed to rebuild initramfs for $KERNEL_VERSION" >&2
    fi
fi
