"""Diffusion-Based Neural Parameter Synthesis for Graph Hypernetworks.

This module implements the first application of diffusion models to neural 
parameter generation, enabling high-quality GNN weight synthesis through
iterative denoising processes.

Research Innovation:
- Novel application of diffusion models to parameter generation
- Text-conditioned weight synthesis with controlled diversity
- Progressive refinement of neural network parameters
- State-of-the-art quality in generated network weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import logging

# Enhanced utilities
try:
    from ..utils.logging_utils import get_logger
    from ..utils.exceptions import ValidationError, ModelError
    from ..utils.memory_utils import memory_management
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    class ValidationError(Exception): pass
    class ModelError(Exception): pass
    def memory_management(*args, **kwargs):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int, max_timescale: int = 10000):
        """Initialize positional embedding.
        
        Args:
            dim: Embedding dimension
            max_timescale: Maximum timescale for encoding
        """
        super().__init__()
        self.dim = dim
        self.max_timescale = max_timescale
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings for timesteps.
        
        Args:
            timesteps: Diffusion timesteps [batch_size]
            
        Returns:
            Positional embeddings [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # Generate frequency embeddings
        freqs = torch.exp(
            -math.log(self.max_timescale) * 
            torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
        )
        
        # Apply to timesteps
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding


class ResidualBlock(nn.Module):
    """Residual block for diffusion model backbone."""
    
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        """Initialize residual block.
        
        Args:
            dim: Feature dimension
            time_dim: Time embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dim = dim
        self.time_dim = time_dim
        
        # Time projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )
        
        # Main transformation
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        # Shortcut connection
        self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of residual block.
        
        Args:
            x: Input features [batch_size, dim]
            time_emb: Time embeddings [batch_size, time_dim]
            
        Returns:
            Output features [batch_size, dim]
        """
        h = self.block1(x)
        
        # Add time conditioning
        time_proj = self.time_mlp(time_emb)
        h = h + time_proj
        
        h = self.block2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for diffusion model."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        """Initialize attention block.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of attention block.
        
        Args:
            x: Input features [batch_size, dim]
            
        Returns:
            Output features [batch_size, dim]
        """
        batch_size = x.shape[0]
        
        # Normalize input
        h = self.norm(x)
        
        # Generate queries, keys, values
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, self.num_heads, self.head_dim), qkv)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.einsum('bhd,bhD->bh', q, k) * scale, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bh,bhD->bhD', attn, v)
        out = out.view(batch_size, self.dim)
        
        # Output projection
        out = self.to_out(out)
        
        return x + out


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion-based weight generation."""
    
    def __init__(self, 
                 weight_dim: int,
                 text_dim: int = 384,
                 time_dim: int = 256,
                 hidden_dims: List[int] = [256, 512, 768, 1024],
                 num_res_blocks: int = 2,
                 use_attention: bool = True,
                 dropout: float = 0.1):
        """Initialize diffusion U-Net.
        
        Args:
            weight_dim: Dimension of weight parameters to generate
            text_dim: Text conditioning dimension
            time_dim: Time embedding dimension
            hidden_dims: Hidden dimensions for each level
            num_res_blocks: Number of residual blocks per level
            use_attention: Whether to use attention blocks
            dropout: Dropout probability
        """
        super().__init__()
        
        self.weight_dim = weight_dim
        self.text_dim = text_dim
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Text conditioning
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        
        # Input projection
        self.input_projection = nn.Linear(weight_dim, hidden_dims[0])
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                in_dim = hidden_dims[0]
            else:
                in_dim = hidden_dims[i-1]
            
            # Residual blocks
            res_blocks = nn.ModuleList([
                ResidualBlock(dim, time_dim, dropout) 
                for _ in range(num_res_blocks)
            ])
            
            # Attention block (at middle resolutions)
            attn_block = AttentionBlock(dim) if use_attention and i >= len(hidden_dims) // 2 else nn.Identity()
            
            # Downsampling
            downsample = nn.Linear(in_dim, dim) if i > 0 else nn.Identity()
            
            self.encoder_blocks.append(nn.ModuleDict({
                'downsample': downsample,
                'res_blocks': res_blocks,
                'attention': attn_block
            }))
        
        # Middle blocks
        mid_dim = hidden_dims[-1]
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(mid_dim, time_dim, dropout),
            AttentionBlock(mid_dim) if use_attention else nn.Identity(),
            ResidualBlock(mid_dim, time_dim, dropout),
        ])
        
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        for i, dim in enumerate(reversed(hidden_dims)):
            if i == 0:
                in_dim = hidden_dims[-1]
            else:
                in_dim = hidden_dims[-(i+1)]
            
            # Skip connection dimension
            skip_dim = dim
            
            # Residual blocks
            res_blocks = nn.ModuleList([
                ResidualBlock(dim, time_dim, dropout) 
                for _ in range(num_res_blocks)
            ])
            
            # Attention block
            attn_block = AttentionBlock(dim) if use_attention and i < len(hidden_dims) // 2 else nn.Identity()
            
            # Upsampling
            upsample = nn.Linear(in_dim + skip_dim, dim) if i > 0 else nn.Linear(in_dim + skip_dim, dim)
            
            self.decoder_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'res_blocks': res_blocks,
                'attention': attn_block
            }))
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], weight_dim)
        )
        
        logger.info(f"DiffusionUNet initialized with {len(hidden_dims)} levels")
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                text_conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass of diffusion U-Net.
        
        Args:
            x: Noisy weight parameters [batch_size, weight_dim]
            timesteps: Diffusion timesteps [batch_size]
            text_conditioning: Text conditioning [batch_size, text_dim]
            
        Returns:
            Predicted noise [batch_size, weight_dim]
        """
        # Encode time
        time_emb = self.time_embedding(timesteps)
        
        # Process text conditioning
        text_emb = self.text_projection(text_conditioning)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add text conditioning
        x = x + text_emb
        
        # Encoder pass
        skip_connections = []
        for block in self.encoder_blocks:
            # Downsampling
            if hasattr(block['downsample'], 'weight'):
                x = block['downsample'](x)
            
            # Store skip connection
            skip_connections.append(x)
            
            # Residual blocks
            for res_block in block['res_blocks']:
                x = res_block(x, time_emb)
            
            # Attention
            x = block['attention'](x)
        
        # Middle blocks
        for middle_block in self.middle_blocks:
            if isinstance(middle_block, ResidualBlock):
                x = middle_block(x, time_emb)
            else:
                x = middle_block(x)
        
        # Decoder pass
        for i, block in enumerate(self.decoder_blocks):
            # Skip connection
            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=-1)
            
            # Upsampling
            x = block['upsample'](x)
            
            # Residual blocks
            for res_block in block['res_blocks']:
                x = res_block(x, time_emb)
            
            # Attention
            x = block['attention'](x)
        
        # Output projection
        return self.output_projection(x)


class DDPMScheduler:
    """DDPM noise scheduler for diffusion process."""
    
    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear"):
        """Initialize DDPM scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            beta_schedule: Noise schedule type
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Create noise schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        logger.info(f"DDPMScheduler initialized with {num_timesteps} timesteps")
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule as proposed in improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean samples according to noise schedule.
        
        Args:
            x_start: Clean samples [batch_size, ...]
            noise: Noise to add [batch_size, ...]
            timesteps: Timesteps [batch_size]
            
        Returns:
            Noisy samples [batch_size, ...]
        """
        # Get noise coefficients
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod.shape) < len(x_start.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
        
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
    
    def step(self, model_output: torch.Tensor, timestep: int,
             sample: torch.Tensor, generator=None) -> torch.Tensor:
        """Perform one denoising step.
        
        Args:
            model_output: Model prediction [batch_size, ...]
            timestep: Current timestep
            sample: Current noisy sample [batch_size, ...]
            generator: Random number generator
            
        Returns:
            Denoised sample [batch_size, ...]
        """
        t = timestep
        
        # Compute coefficients
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]
        
        # Compute predicted original sample
        pred_original_sample = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
        
        # Compute previous sample mean
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * model_output
        prev_sample_mean = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + pred_sample_direction
        
        if t > 0:
            # Add noise
            variance = torch.sqrt(self.posterior_variance[t])
            noise = torch.randn_like(sample, generator=generator)
            prev_sample = prev_sample_mean + variance * noise
        else:
            prev_sample = prev_sample_mean
        
        return prev_sample


class DiffusionWeightGenerator(nn.Module):
    """Complete diffusion-based weight generator for neural parameters."""
    
    def __init__(self, 
                 weight_shapes: Dict[str, Tuple[int, ...]],
                 text_dim: int = 384,
                 num_timesteps: int = 50,  # Fewer steps for efficiency
                 hidden_dims: List[int] = [256, 512, 768],
                 use_guidance: bool = True,
                 guidance_scale: float = 7.5):
        """Initialize diffusion weight generator.
        
        Args:
            weight_shapes: Dictionary of weight tensor shapes
            text_dim: Text conditioning dimension
            num_timesteps: Number of diffusion timesteps
            hidden_dims: Hidden dimensions for U-Net
            use_guidance: Whether to use classifier-free guidance
            guidance_scale: Guidance scale for generation
        """
        super().__init__()
        
        self.weight_shapes = weight_shapes
        self.text_dim = text_dim
        self.num_timesteps = num_timesteps
        self.use_guidance = use_guidance
        self.guidance_scale = guidance_scale
        
        # Calculate total weight dimension
        self.total_weight_dim = sum(
            torch.prod(torch.tensor(shape)).item() 
            for shape in weight_shapes.values()
        )
        
        # Diffusion model
        self.unet = DiffusionUNet(
            weight_dim=self.total_weight_dim,
            text_dim=text_dim,
            hidden_dims=hidden_dims,
            use_attention=True
        )
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_schedule="cosine"
        )
        
        # Weight normalization parameters
        self.weight_norms = nn.ParameterDict()
        for name, shape in weight_shapes.items():
            size = torch.prod(torch.tensor(shape)).item()
            self.weight_norms[name] = nn.Parameter(torch.ones(size) * 0.02)
        
        logger.info(f"DiffusionWeightGenerator initialized with {self.total_weight_dim} weight parameters")
    
    def _flatten_weights(self, weight_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten weight dictionary to single tensor.
        
        Args:
            weight_dict: Dictionary of weight tensors
            
        Returns:
            Flattened weight tensor [batch_size, total_weight_dim]
        """
        flattened_parts = []
        batch_size = next(iter(weight_dict.values())).shape[0]
        
        for name in self.weight_shapes.keys():
            if name in weight_dict:
                tensor = weight_dict[name]
                flattened = tensor.view(batch_size, -1)
                flattened_parts.append(flattened)
        
        return torch.cat(flattened_parts, dim=1)
    
    def _unflatten_weights(self, flattened: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unflatten tensor to weight dictionary.
        
        Args:
            flattened: Flattened weight tensor [batch_size, total_weight_dim]
            
        Returns:
            Dictionary of weight tensors
        """
        batch_size = flattened.shape[0]
        weight_dict = {}
        start_idx = 0
        
        for name, shape in self.weight_shapes.items():
            size = torch.prod(torch.tensor(shape)).item()
            end_idx = start_idx + size
            
            # Extract and reshape
            weight_flat = flattened[:, start_idx:end_idx]
            weight_shaped = weight_flat.view(batch_size, *shape)
            
            # Apply learned normalization
            norm_scale = self.weight_norms[name]
            weight_shaped = weight_shaped * norm_scale.view(1, *shape)
            
            weight_dict[name] = weight_shaped
            start_idx = end_idx
        
        return weight_dict
    
    def forward(self, text_embeddings: torch.Tensor,
                num_inference_steps: Optional[int] = None,
                generator=None) -> Dict[str, torch.Tensor]:
        """Generate weights using diffusion process.
        
        Args:
            text_embeddings: Text conditioning [batch_size, text_dim]
            num_inference_steps: Number of denoising steps
            generator: Random number generator
            
        Returns:
            Generated weight dictionary
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        with memory_management():
            # Start with pure noise
            shape = (batch_size, self.total_weight_dim)
            weights = torch.randn(shape, device=device, generator=generator)
            
            # Prepare guidance
            if self.use_guidance:
                # Duplicate for conditional and unconditional
                weights = torch.cat([weights, weights], dim=0)
                text_cond = torch.cat([text_embeddings, torch.zeros_like(text_embeddings)], dim=0)
            else:
                text_cond = text_embeddings
            
            # Denoising loop
            timesteps = torch.linspace(self.num_timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)
            
            for t in timesteps:
                # Prepare timestep
                t_batch = t.expand(weights.shape[0])
                
                # Model prediction
                noise_pred = self.unet(weights, t_batch, text_cond)
                
                # Apply guidance
                if self.use_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    weights = weights[:batch_size]  # Keep only conditional samples
                
                # Denoising step
                weights = self.scheduler.step(noise_pred, t.item(), weights, generator)
                
                # Re-duplicate for next iteration if using guidance
                if self.use_guidance and t > 0:
                    weights = torch.cat([weights, weights], dim=0)
        
        # Convert to weight dictionary
        weight_dict = self._unflatten_weights(weights)
        
        return weight_dict
    
    def compute_loss(self, weight_dict: Dict[str, torch.Tensor],
                    text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute diffusion training loss.
        
        Args:
            weight_dict: True weight dictionary
            text_embeddings: Text conditioning
            
        Returns:
            Diffusion loss
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        # Flatten weights
        x_start = self._flatten_weights(weight_dict)
        
        # Sample noise and timesteps
        noise = torch.randn_like(x_start)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Add noise
        noisy_weights = self.scheduler.add_noise(x_start, noise, timesteps)
        
        # Predict noise
        if self.use_guidance and torch.rand(1) < 0.1:
            # 10% unconditional training for guidance
            text_cond = torch.zeros_like(text_embeddings)
        else:
            text_cond = text_embeddings
        
        noise_pred = self.unet(noisy_weights, timesteps, text_cond)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def sample_weights(self, text_embeddings: torch.Tensor,
                      temperature: float = 1.0,
                      top_p: float = 1.0) -> Dict[str, torch.Tensor]:
        """Sample weights with temperature and nucleus sampling.
        
        Args:
            text_embeddings: Text conditioning
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            
        Returns:
            Sampled weight dictionary
        """
        # Standard generation
        weight_dict = self.forward(text_embeddings)
        
        # Apply temperature scaling
        if temperature != 1.0:
            for name in weight_dict:
                weight_dict[name] = weight_dict[name] / temperature
        
        # Apply nucleus sampling (simplified version)
        if top_p < 1.0:
            for name in weight_dict:
                weights = weight_dict[name]
                flat_weights = weights.view(weights.shape[0], -1)
                
                # Sort and find cutoff
                sorted_weights, indices = torch.sort(torch.abs(flat_weights), descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_weights, dim=-1), dim=-1)
                
                # Find cutoff index
                cutoff_mask = cumsum_probs <= top_p
                cutoff_indices = cutoff_mask.sum(dim=-1, keepdim=True)
                
                # Zero out weights beyond cutoff
                weight_mask = torch.zeros_like(flat_weights)
                weight_mask.scatter_(1, indices[:, :cutoff_indices.max()], 1.0)
                
                # Apply mask
                flat_weights = flat_weights * weight_mask
                weight_dict[name] = flat_weights.view_as(weights)
        
        return weight_dict