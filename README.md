# Face Generation using VQVAE, DDPM, DiT and DDIM

## Overview

This project implements a complete latent diffusion pipeline for high-quality face generation using the CelebA-HQ dataset.

Instead of performing diffusion directly on high-resolution images, a **Vector Quantized Variational Autoencoder (VQVAE)** is first trained to compress images into a discrete latent space. Diffusion models are then trained on these latent representations, significantly reducing computational requirements while maintaining image quality.

### Models Implemented

- VQVAE
- DDPM (Denoising Diffusion Probabilistic Model)
- DiT (Diffusion Transformer)
- DDIM Sampling

---

## Dataset

### CelebA-HQ

- Resolution: 256 × 256
- RGB Images
- Human Face Dataset

### Preprocessing

- Center Crop
- Data Augmentation
- Resize to 256 × 256
- Convert to Tensor
- Normalize to [-1,1]

---

## Complete Pipeline

```text
Input Image
      │
      ▼
VQVAE Encoder
      │
      ▼
Latent Representation
      │
      ▼
Vector Quantization
      │
      ▼
Quantized Latent
      │
      ▼
DDPM / DiT
(Diffusion Model)
      │
      ▼
Denoised Latent
      │
      ▼
VQVAE Decoder
      │
      ▼
Generated Face
```

---

# Stage 1: VQVAE

## Architecture

```text
Image
  │
  ▼
Encoder
  │
  ▼
Latent Features
  │
  ▼
Codebook Quantization
  │
  ▼
Decoder
  │
  ▼
Reconstructed Image
```

### Key Components

- Encoder
- Codebook Vectors
- Vector Quantization
- Decoder
- Straight Through Estimator
- LPIPS Loss
- PatchGAN Loss

---

## VQVAE Loss

$$
L =
L_{rec}
+
L_{codebook}
+
L_{commit}
+
L_{LPIPS}
+
L_{adv}
$$

### VQVAE Parameters

| Parameter | Value |
|------------|---------|
| Image Size | 256 |
| Batch Size | 4 |
| Epochs | 20 |
| Learning Rate | 1e-5 |

---

# Stage 2: DDPM

## Forward Diffusion

Noise is gradually added to latent vectors.

$$
q(x_t|x_{t-1}) = 
\mathcal{N}
(
\sqrt{\alpha_t}x_{t-1},
\beta_t I
)
$$

---

## Reparameterization

The noisy latent at timestep t can be obtained directly using:

$$
x_t=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

where

$$
\epsilon \sim \mathcal{N}(0,I)
$$

---

## Training Objective

The diffusion model predicts the added noise.

$$
L=
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
$$

### DDPM Parameters

| Parameter | Value |
|------------|---------|
| Timesteps | 1000 |
| Batch Size | 64 |
| Epochs | 100 |
| Learning Rate | 1e-4 |

---

# Stage 3: Latent Diffusion

Instead of applying diffusion directly to images, diffusion is performed in the VQVAE latent space.

### Advantages

- Faster Training
- Lower GPU Memory
- Faster Sampling
- Better Scalability
- High Quality Image Generation

---

# Stage 4: Diffusion Transformer (DiT)

Traditional diffusion models use U-Net for noise prediction.

In this project, a Diffusion Transformer (DiT) is implemented to replace the U-Net.

---

## DiT Architecture

```text
Noisy Latent
      │
      ▼
Patch Embedding
      │
      ▼
Transformer Encoder Blocks
      │
      ▼
Noise Prediction
```

### Components

- Patch Embedding
- Layer Normalization
- Multi-Head Self Attention
- Feed Forward Network
- Adaptive LayerNorm Zero (AdaLN-Zero)

---

## Multi-Head Attention

$$
Attention(Q,K,V) = 
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

### DiT Parameters

| Parameter | Value |
|------------|---------|
| Hidden Size | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Patch Size | 2 |
| Batch Size | 128 |
| Timesteps | 1000 |

---

# Stage 5: DDIM Sampling

DDIM accelerates image generation by reducing the number of reverse diffusion steps.

---

## Predicted Clean Sample

$$
\hat{x}_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}
{\sqrt{\bar{\alpha}_t}}
$$

---

## DDIM Update Equation

$$
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0
+
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\hat{\epsilon}
+
\sigma_t z
$$

where

$$
z \sim \mathcal{N}(0,I)
$$

For deterministic DDIM, η = 0 and σt = 0, reducing the update rule to:

$$
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0
+
\sqrt{1-\bar{\alpha}_{t-1}}\hat{\epsilon}
$$

Using DDIM significantly reduces inference time compared to DDPM while maintaining image quality.

---

## Face Generation Comparison

| Model | Quality | Sampling Speed |
|---------|---------|---------|
| DDPM | High | Slow |
| DiT | Very High | Slow |
| DDIM | High | Fast |

---

## Generated Samples

Add generated images here.

```markdown
![Generated Face](results/generated_face.png)
```

---

# Future Work

- Text-to-Image Conditioning
- Class Conditional Generation
- Larger Codebooks
- Higher Resolution Face Synthesis
- Faster Diffusion Schedulers

---

# References

1. VQ-VAE: Neural Discrete Representation Learning
2. Denoising Diffusion Probabilistic Models
3. Diffusion Models Beat GANs on Image Synthesis
4. High-Resolution Image Synthesis with Latent Diffusion Models
5. Scalable Diffusion Models with Transformers
---

## Author

**Pranav Deshpande**  
IIT Jodhpur  
Deep Learning • Generative AI • Diffusion Models
