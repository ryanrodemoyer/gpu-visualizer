// GPU spec sheet values. VRAM is in GiB (GPU memory is binary-sized), bandwidth in GB/s (decimal).
// Every row carries a source so numbers can be checked.

export type Vendor = 'NVIDIA' | 'AMD'
export type Segment = 'consumer' | 'workstation' | 'datacenter'

export interface Gpu {
  id: string
  name: string
  vendor: Vendor
  segment: Segment
  architecture: string
  vramGiB: number
  memoryType: string
  busWidthBits: number
  /** Peak memory bandwidth, GB/s (1 GB = 1e9 bytes) */
  bandwidthGBs: number
  tdpW: number
  launchPriceUsd?: number
  released: string
  source: string
}

const tpu = (q: string) => `https://www.techpowerup.com/gpu-specs/?q=${encodeURIComponent(q)}`

export const GPUS: Gpu[] = [
  { id: 'rtx-3060-12', name: 'RTX 3060 12GB', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ampere', vramGiB: 12, memoryType: 'GDDR6', busWidthBits: 192, bandwidthGBs: 360, tdpW: 170, launchPriceUsd: 329, released: '2021-02', source: tpu('RTX 3060') },
  { id: 'rtx-3090', name: 'RTX 3090', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ampere', vramGiB: 24, memoryType: 'GDDR6X', busWidthBits: 384, bandwidthGBs: 936, tdpW: 350, launchPriceUsd: 1499, released: '2020-09', source: 'https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622' },
  { id: 'rtx-3090-ti', name: 'RTX 3090 Ti', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ampere', vramGiB: 24, memoryType: 'GDDR6X', busWidthBits: 384, bandwidthGBs: 1008, tdpW: 450, launchPriceUsd: 1999, released: '2022-03', source: tpu('RTX 3090 Ti') },
  { id: 'rtx-4060-ti-16', name: 'RTX 4060 Ti 16GB', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ada Lovelace', vramGiB: 16, memoryType: 'GDDR6', busWidthBits: 128, bandwidthGBs: 288, tdpW: 165, launchPriceUsd: 499, released: '2023-07', source: tpu('RTX 4060 Ti 16 GB') },
  { id: 'rtx-4070-ti-super', name: 'RTX 4070 Ti SUPER', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ada Lovelace', vramGiB: 16, memoryType: 'GDDR6X', busWidthBits: 256, bandwidthGBs: 672, tdpW: 285, launchPriceUsd: 799, released: '2024-01', source: tpu('RTX 4070 Ti SUPER') },
  { id: 'rtx-4080', name: 'RTX 4080', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ada Lovelace', vramGiB: 16, memoryType: 'GDDR6X', busWidthBits: 256, bandwidthGBs: 716.8, tdpW: 320, launchPriceUsd: 1199, released: '2022-11', source: tpu('RTX 4080') },
  { id: 'rtx-4090', name: 'RTX 4090', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Ada Lovelace', vramGiB: 24, memoryType: 'GDDR6X', busWidthBits: 384, bandwidthGBs: 1008, tdpW: 450, launchPriceUsd: 1599, released: '2022-10', source: tpu('RTX 4090') },
  { id: 'rtx-5060-ti-16', name: 'RTX 5060 Ti 16GB', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Blackwell', vramGiB: 16, memoryType: 'GDDR7', busWidthBits: 128, bandwidthGBs: 448, tdpW: 180, launchPriceUsd: 429, released: '2025-04', source: tpu('RTX 5060 Ti 16 GB') },
  { id: 'rtx-5070-ti', name: 'RTX 5070 Ti', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Blackwell', vramGiB: 16, memoryType: 'GDDR7', busWidthBits: 256, bandwidthGBs: 896, tdpW: 300, launchPriceUsd: 749, released: '2025-02', source: tpu('RTX 5070 Ti') },
  { id: 'rtx-5080', name: 'RTX 5080', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Blackwell', vramGiB: 16, memoryType: 'GDDR7', busWidthBits: 256, bandwidthGBs: 960, tdpW: 360, launchPriceUsd: 999, released: '2025-01', source: tpu('RTX 5080') },
  { id: 'rtx-5090', name: 'RTX 5090', vendor: 'NVIDIA', segment: 'consumer', architecture: 'Blackwell', vramGiB: 32, memoryType: 'GDDR7', busWidthBits: 512, bandwidthGBs: 1792, tdpW: 575, launchPriceUsd: 1999, released: '2025-01', source: 'https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216' },
  { id: 'rx-7900-xtx', name: 'Radeon RX 7900 XTX', vendor: 'AMD', segment: 'consumer', architecture: 'RDNA 3', vramGiB: 24, memoryType: 'GDDR6', busWidthBits: 384, bandwidthGBs: 960, tdpW: 355, launchPriceUsd: 999, released: '2022-12', source: tpu('RX 7900 XTX') },
  { id: 'rtx-a6000', name: 'RTX A6000', vendor: 'NVIDIA', segment: 'workstation', architecture: 'Ampere', vramGiB: 48, memoryType: 'GDDR6', busWidthBits: 384, bandwidthGBs: 768, tdpW: 300, launchPriceUsd: 4650, released: '2020-10', source: tpu('RTX A6000') },
  { id: 'rtx-6000-ada', name: 'RTX 6000 Ada', vendor: 'NVIDIA', segment: 'workstation', architecture: 'Ada Lovelace', vramGiB: 48, memoryType: 'GDDR6', busWidthBits: 384, bandwidthGBs: 960, tdpW: 300, launchPriceUsd: 6800, released: '2022-12', source: tpu('RTX 6000 Ada') },
  { id: 'rtx-pro-6000', name: 'RTX PRO 6000 Blackwell', vendor: 'NVIDIA', segment: 'workstation', architecture: 'Blackwell', vramGiB: 96, memoryType: 'GDDR7', busWidthBits: 512, bandwidthGBs: 1792, tdpW: 600, released: '2025-03', source: 'https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/' },
  { id: 'l40s', name: 'L40S', vendor: 'NVIDIA', segment: 'datacenter', architecture: 'Ada Lovelace', vramGiB: 48, memoryType: 'GDDR6', busWidthBits: 384, bandwidthGBs: 864, tdpW: 350, released: '2023-08', source: tpu('L40S') },
  { id: 'a100-80-sxm', name: 'A100 80GB SXM', vendor: 'NVIDIA', segment: 'datacenter', architecture: 'Ampere', vramGiB: 80, memoryType: 'HBM2e', busWidthBits: 5120, bandwidthGBs: 2039, tdpW: 400, released: '2020-11', source: tpu('A100 SXM4 80 GB') },
  { id: 'h100-pcie', name: 'H100 80GB PCIe', vendor: 'NVIDIA', segment: 'datacenter', architecture: 'Hopper', vramGiB: 80, memoryType: 'HBM2e', busWidthBits: 5120, bandwidthGBs: 2039, tdpW: 350, released: '2022-03', source: tpu('H100 PCIe 80 GB') },
  { id: 'h100-sxm', name: 'H100 80GB SXM', vendor: 'NVIDIA', segment: 'datacenter', architecture: 'Hopper', vramGiB: 80, memoryType: 'HBM3', busWidthBits: 5120, bandwidthGBs: 3352, tdpW: 700, released: '2022-03', source: tpu('H100 SXM5 80 GB') },
]

export const gpuById = (id: string) => GPUS.find(g => g.id === id)
