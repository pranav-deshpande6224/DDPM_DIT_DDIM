# Face Generation using VQVAE, DDPM, DiT and DDIM

## Overview

This project implements a complete latent diffusion pipeline for high-quality face generation using the CelebA-HQ dataset.

Instead of performing diffusion directly on high-resolution images, a Vector Quantized Variational Autoencoder (VQVAE) is first trained to compress images into a discrete latent space. Diffusion models are then trained on these compact latent representations, significantly reducing computational requirements while maintaining image quality.

The project explores three major components:

- VQVAE for latent representation learning
- DDPM for diffusion-based image generation
- DiT (Diffusion Transformer) for transformer-based diffusion
- DDIM for accelerated sampling

---

## Pipeline

```text
Input Image
     │
     ▼
┌─────────────┐
│   VQVAE     │
│  Encoder    │
└─────────────┘
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
 ┌───────────────┐
 │ DDPM / DiT    │
 │ Diffusion     │
 └───────────────┘
     │
     ▼
Denoised Latent
     │
     ▼
┌─────────────┐
│   VQVAE     │
│  Decoder    │
└─────────────┘
     │
     ▼
Generated Face
```

---

# Dataset

## CelebA-HQ

CelebA-HQ is a high-quality face dataset consisting of celebrity face images.

### Preprocessing

- Center Crop
- Data Augmentation
- Resize Images
- Convert to Tensor
- Normalize Pixel Values to [-1,1]

### Image Resolution

```text
256 × 256 × 3
```

---

# Stage 1 : VQVAE

## Motivation

Traditional Variational Autoencoders compress images into continuous latent representations.

VQVAE introduces a discrete latent space using a learnable codebook. Each encoder output vector is replaced by its nearest codebook vector, allowing the model to learn meaningful discrete representations.

---

## Architecture

```text
Image
  │
  ▼
Encoder
  │
  ▼
Latent Feature Map
  │
  ▼
Vector Quantization
  │
  ▼
Codebook Vector
  │
  ▼
Decoder
  │
  ▼
Reconstructed Image
```

---

## Vector Quantization

The encoder produces a latent representation:

\[
z_e(x)
\]

For every latent vector, the nearest codebook embedding is selected using Euclidean distance.

\[
z_q(x)=argmin_j ||z_e(x)-e_j||
\]

where

- \(e_j\) is a codebook vector
- \(z_q(x)\) is the quantized latent representation

---

## Reconstruction Loss

The reconstructed image should be as close as possible to the original image.

\[
L_{rec}
=
||x-\hat{x}||^2
\]

---

## Codebook Loss

Updates the codebook vectors.

\[
L_{codebook}
=
||sg[z_e(x)]-e||^2
\]

where

\[
sg(\cdot)
\]

denotes the stop-gradient operator.

---

## Commitment Loss

Encourages encoder outputs to stay close to selected codebook vectors.

\[
L_{commit}
=
||z_e(x)-sg[e]||^2
\]

---

## Straight Through Estimator

Vector quantization is non-differentiable.

To allow gradient flow through the encoder, a Straight Through Estimator (STE) is used.

```python
quantized =
encoder_output +
(quantized - encoder_output).detach()
```

This allows gradients to bypass the quantization operation.

---

## LPIPS Perceptual Loss

Pixel-wise losses often produce blurry reconstructions.

To improve visual quality, LPIPS (Learned Perceptual Image Patch Similarity) is used.

Both original and reconstructed images are passed through a pretrained network.

\[
d_l
=
||\phi_l(x)-\phi_l(\hat{x})||^2
\]

LPIPS:

\[
L_{LPIPS}
=
\sum_l w_l d_l
\]

where:

- \(\phi_l\) = feature map at layer \(l\)
- \(w_l\) = learned LPIPS weights

---

## PatchGAN Adversarial Loss

A PatchGAN discriminator is used to improve reconstruction realism.

Instead of classifying the entire image, PatchGAN classifies local image patches.

```text
Image
 │
 ▼
Conv
 │
 ▼
Conv
 │
 ▼
Conv
 │
 ▼
Patch Predictions
```

---

## Final VQVAE Loss

\[
L =
L_{rec}
+
\lambda_{cb}L_{codebook}
+
\lambda_{commit}L_{commit}
+
\lambda_{LPIPS}L_{LPIPS}
+
\lambda_{adv}L_{adv}
\]

---

## VQVAE Parameters

| Parameter | Value |
|------------|---------|
| Image Size | 256 |
| Batch Size | 4 |
| Epochs | 20 |
| Learning Rate | 1e-5 |

---

# Stage 2 : DDPM

## Motivation

After learning compact latent representations, diffusion is performed inside latent space.

This greatly reduces computational cost compared to pixel-space diffusion.

---

## Forward Diffusion Process

Noise is gradually added to latent vectors.

\[
q(x_t|x_{t-1})
=
\mathcal N
(
\sqrt{\alpha_t}x_{t-1},
\beta_tI
)
\]

where

\[
\alpha_t = 1-\beta_t
\]

---

## Reparameterization Form

After repeatedly applying the forward process:

\[
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
\]

where

\[
\epsilon \sim \mathcal N(0,I)
\]

and

\[
\bar{\alpha}_t
=
\prod_{i=1}^{t}\alpha_i
\]

---

## Noise Schedule

Linear schedule:

\[
\beta_1=10^{-4}
\]

\[
\beta_T=2\times10^{-2}
\]

\[
T=1000
\]

---

## Reverse Diffusion

The diffusion model predicts the added noise.

\[
\epsilon_\theta(x_t,t)
\]

The model learns how to remove noise progressively.

---

## Training Objective

The DDPM loss becomes:

\[
L
=
E
\left[
||\epsilon-\epsilon_\theta(x_t,t)||^2
\right]
\]

which is simply Mean Squared Error between actual and predicted noise.

---

## DDPM Training Parameters

| Parameter | Value |
|------------|---------|
| Timesteps | 1000 |
| Batch Size | 64 |
| Epochs | 100 |
| Learning Rate | 1e-4 |

---

# Stage 3 : Latent Diffusion Model

Instead of diffusing images:

```text
Image
  │
  ▼
VQVAE Encoder
  │
  ▼
Latent Space
  │
  ▼
Diffusion Process
  │
  ▼
Latent Space
  │
  ▼
VQVAE Decoder
  │
  ▼
Generated Image
```

Benefits:

- Faster Training
- Lower Memory Usage
- Better Scalability
- High Quality Generation

---

# Stage 4 : Diffusion Transformer (DiT)

## Motivation

Traditional diffusion models use U-Net.

DiT replaces the U-Net with a Transformer architecture for noise prediction.

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

---

## Components

### Patch Embedding

Latent representations are divided into patches and projected into embeddings.

### Layer Normalization

\[
\hat{x}
=
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
\]

---

## Multi Head Self Attention

Given input:

\[
X
\]

Query, Key and Value projections:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

Attention:

\[
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)V
\]

---

## Adaptive LayerNorm Zero

DiT uses Adaptive LayerNorm Zero (AdaLN-Zero) conditioning.

Benefits:

- Stable Training
- Better Time-Step Conditioning
- Improved Image Quality

---

## DiT Parameters

| Parameter | Value |
|------------|---------|
| Timesteps | 1000 |
| Learning Rate | 1e-5 |
| Hidden Size | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Patch Size | 2 |
| Batch Size | 128 |

---

# Stage 5 : DDIM Sampling

## Motivation

DDPM requires all 1000 reverse diffusion steps.

DDIM accelerates generation using a deterministic sampling process.

---

## Predicted Clean Sample

Using predicted noise:

\[
\hat{x}_0
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}
{\sqrt{\bar{\alpha}_t}}
\]

---

## DDIM Variance

\[
\sigma_t
=
\eta
\sqrt{
\frac{1-\bar{\alpha}_{t-1}}
{1-\bar{\alpha}_t}
}
\sqrt{
1-
\frac{\bar{\alpha}_t}
{\bar{\alpha}_{t-1}}
}
\]

---

## DDPM Direction Term

\[
d_t
=
\sqrt{
1-\bar{\alpha}_{t-1}
-\sigma_t^2
}
\hat{\epsilon}
\]

---

## DDIM Update Equation

\[
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}
\hat{x}_0
+
d_t
+
\sigma_t z
\]

where

\[
z\sim \mathcal N(0,I)
\]

---

## Deterministic DDIM

For

\[
\eta = 0
\]

\[
\sigma_t=0
\]

and the process becomes deterministic:

\[
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}
\hat{x}_0
+
\sqrt{1-\bar{\alpha}_{t-1}}
\hat{\epsilon}
\]

This significantly reduces inference time.

---

# Results

## Reconstruction Results

| Metric | Score |
|----------|----------|
| PSNR | Add Result |
| SSIM | Add Result |
| LPIPS | Add Result |

---

## Face Generation Comparison

| Model | Image Quality | Sampling Speed |
|---------|---------|---------|
| DDPM | High | Slow |
| DiT | Very High | Slow |
| DDIM | High | Fast |

---

# Future Improvements

- Class Conditional Generation
- Text-to-Image Conditioning
- Larger VQVAE Codebooks
- Faster Sampling Schedulers
- Higher Resolution Face Synthesis

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
Deep Learning | Generative AI | Diffusion Models
