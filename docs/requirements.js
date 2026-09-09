import DocLink from '@site/src/components/DocLink';

const macDetails = (
  <ul>
    <li>macOS Sequoia (15) or later</li>
    <li>Apple silicon (M1 - M5 processor)</li>
    <li>Python 3.10 - 3.14</li>
    <li>Xcode or Xcode Command Line Tools 16 or later</li>
  </ul>
);

const linuxDetails = (
  <ul>
    <li>Ubuntu 22.04 LTS</li>
    <li>x86-64-v3 or later CPU (Haswell-class or newer; CPUs from approximately 2013
  onward; with&nbsp;<a href="https://www.intel.com/content/www/us/en/support/articles/000057621/processors.html" target="_blank" rel="noopener noreferrer">SSE4.2 or
      newer</a>)
      or AWS Graviton2/3 CPU</li>
    <li>Minimum 8 GiB RAM (or much more, depending on the model you run)</li>
    <li>Python 3.10 - 3.14</li>
    <li>g++ or clang++ C++ compiler</li>
  </ul>
);

const wslDetails = (
  <p>Windows is not officially supported.
    You can still try MAX/Mojo on Windows <a
    href="https://learn.microsoft.com/en-us/windows/wsl/install">with WSL</a>,
    using a compatible version of Ubuntu (see the Linux requirements).</p>
);

const gpuDetails = (
  <ul>
    <li>NVIDIA B200, H100, H200, A100, A10, L4, L40, RTX 50XX, RTX 40XX, RTX 30XX
      <ul>
        <li>NVIDIA GPU driver 580 or later</li>
      </ul>
    </li>
    <li>AMD Instinct MI355X, MI300X, MI325X
      <ul>
        <li>AMD GPU driver 6.3.3 or later</li>
      </ul>
    </li>
  </ul>
);

const dockerDetails = (
  <ul>
    <li>Docker and Docker Compose</li>
    <li>If you're using an NVIDIA system, make sure you enable <DocLink
      to="https://docs.docker.com/config/containers/resource_constraints/#gpu"
      >NVIDIA GPU support</DocLink>—in particular, make sure you have the <DocLink
      to="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
      >NVIDIA Container Toolkit</DocLink></li>
  </ul>
);

export const requirementsNoGPU = {
  left: [
    {
      id: 'mac',
      label: 'Mac',
      details: macDetails,
    },
    {
      id: 'linux',
      label: 'Linux',
      details: linuxDetails,
    },
    {
      id: 'wsl',
      label: 'WSL',
      details: wslDetails,
    }
  ]
};

export const requirementsWithDockerAndGPU = {
  left: [
    {
      id: 'linux',
      label: 'Linux',
      details: linuxDetails,
    },
    {
      id: 'wsl',
      label: 'WSL',
      details: wslDetails,
    }
  ],
  right: [
    {
      id: 'nvidia',
      label: 'GPU',
      details: gpuDetails,
    },
    {
      id: 'docker',
      label: 'Docker',
      details: dockerDetails,
    }
  ]
};

export const requirementsWithGPU = {
  left: [
    {
      id: 'mac',
      label: 'Mac',
      details: macDetails,
    },
    {
      id: 'linux',
      label: 'Linux',
      details: linuxDetails,
    },
    {
      id: 'wsl',
      label: 'WSL',
      details: wslDetails,
    }
  ],
  right: [
    {
      id: 'nvidia',
      label: 'GPU',
      details: gpuDetails,
    }
  ]
};

export const requirementsNoMacWithGPU = {
  left: [
    {
      id: 'linux',
      label: 'Linux',
      details: linuxDetails,
    },
    {
      id: 'wsl',
      label: 'WSL',
      details: wslDetails,
    }
  ],
  right: [
    {
      id: 'nvidia',
      label: 'GPU',
      details: gpuDetails,
    }
  ]
};
