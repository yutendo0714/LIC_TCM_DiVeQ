import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class CodebookReplacement:
    """
    Improved codebook replacement strategy (Appendix B.1)
    Uses importance sampling based on codeword occurrence probability
    """
    
    def __init__(
        self,
        num_embeddings: int,
        discard_threshold: float = 0.01,
        replacement_interval: int = 100,
    ):
        self.num_embeddings = num_embeddings
        self.discard_threshold = discard_threshold
        self.replacement_interval = replacement_interval
        self.usage_counter = torch.zeros(num_embeddings)
        self.iteration = 0
        
    def update_usage(self, indices: torch.Tensor):
        """Update codeword usage statistics"""
        unique, counts = torch.unique(indices, return_counts=True)
        for idx, count in zip(unique, counts):
            self.usage_counter[idx.item()] += count.item()
        self.iteration += 1
    
    def replace_unused(self, codebook_weight: torch.Tensor) -> torch.Tensor:
        """
        Replace unused codewords with perturbations of used ones
        Uses importance sampling based on usage probability
        """
        if self.iteration % self.replacement_interval != 0:
            return codebook_weight
        
        # Compute usage probabilities
        total_usage = self.usage_counter.sum()
        if total_usage == 0:
            return codebook_weight
        
        usage_prob = self.usage_counter / total_usage
        
        # Find unused codewords (below threshold)
        unused_mask = usage_prob < self.discard_threshold
        used_mask = ~unused_mask
        
        if unused_mask.sum() == 0:
            self.usage_counter.zero_()
            return codebook_weight
        
        # Importance sampling: select from used codewords based on probability
        used_indices = torch.where(used_mask)[0]
        used_probs = usage_prob[used_mask]
        used_probs = used_probs / used_probs.sum()  # Normalize
        
        # Replace unused codewords
        with torch.no_grad():
            for unused_idx in torch.where(unused_mask)[0]:
                # Sample from used codewords with importance sampling
                sampled_idx = used_indices[torch.multinomial(used_probs, 1).item()]
                # Add small perturbation
                perturbation = torch.randn_like(codebook_weight[sampled_idx]) * 0.01
                codebook_weight[unused_idx] = codebook_weight[sampled_idx] + perturbation
        
        # Reset counter after replacement
        self.usage_counter.zero_()
        
        return codebook_weight


class DiVeQ(nn.Module):
    """
    Differentiable Vector Quantization (DiVeQ)
    
    DiVeQ treats quantization as adding an error vector that mimics the quantization
    distortion, keeping the forward pass hard while letting gradients flow.
    
    Based on: "DiVeQ: Differentiable Vector Quantization using the Reparameterization Trick"
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sigma_squared: float = 1e-3,
        commitment_cost: float = 0.25,
        use_codebook_replacement: bool = True,
        replacement_config: Optional[Dict] = None,
    ):
        """
        Args:
            num_embeddings: Number of codewords in the codebook (K)
            embedding_dim: Dimension of each codeword (D)
            sigma_squared: Variance for directional noise (σ²)
            commitment_cost: Weight for commitment loss (β)
            use_codebook_replacement: Whether to use codebook replacement
            replacement_config: Config for codebook replacement
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sigma_squared = sigma_squared
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # Codebook replacement (Appendix B.1)
        self.use_codebook_replacement = use_codebook_replacement
        if use_codebook_replacement:
            config = replacement_config or {}
            self.replacement = CodebookReplacement(
                num_embeddings=num_embeddings,
                discard_threshold=config.get('discard_threshold', 0.01),
                replacement_interval=config.get('replacement_interval', 100),
            )
        
    def forward(
        self, 
        z: torch.Tensor,
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with DiVeQ quantization
        
        Args:
            z: Input tensor of shape [B, D, H, W] or [B, L, D]
            return_loss: Whether to return the loss
            
        Returns:
            z_q: Quantized output (differentiable)
            loss: VQ loss (reconstruction + commitment)
            indices: Indices of selected codewords
        """
        # Reshape input for computation
        input_shape = z.shape
        flat_input = z.reshape(-1, self.embedding_dim)  # [N, D]
        
        # Find nearest codeword for each input (hard assignment)
        distances = torch.cdist(flat_input, self.codebook.weight)  # [N, K]
        indices = torch.argmin(distances, dim=1)  # [N]
        c_i_star = self.codebook.weight[indices]  # [N, D]
        
        # Compute directional vector: d = c_i* - z
        d_vec = c_i_star - flat_input  # [N, D]
        
        # Sample directional noise: v_d = v + d, where v ~ N(0, σ²I)
        if self.training:
            v = torch.randn_like(d_vec) * (self.sigma_squared ** 0.5)
            v_d = v + d_vec
        else:
            # At inference, use deterministic mapping (σ² → 0)
            v_d = d_vec
        
        # Compute magnitude: ||c_i* - z||₂
        magnitude = torch.norm(d_vec, p=2, dim=1, keepdim=True)  # [N, 1]
        
        # Compute normalized direction with stop gradient
        direction = v_d / (torch.norm(v_d, p=2, dim=1, keepdim=True) + 1e-8)
        direction = direction.detach()  # Stop gradient
        
        # DiVeQ quantization: z_q = z + ||c_i* - z||₂ · sg(v_d / ||v_d||₂)
        z_q = flat_input + magnitude * direction  # [N, D]
        
        # Reshape back to original shape
        z_q = z_q.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])
        
        # Update codebook usage for replacement
        if self.training and self.use_codebook_replacement:
            self.replacement.update_usage(indices.flatten())
            self.codebook.weight.data = self.replacement.replace_unused(
                self.codebook.weight.data
            )
        
        # Compute loss
        loss = None
        if return_loss:
            # Codebook loss: ||sg[z] - c_i*||²
            codebook_loss = F.mse_loss(c_i_star, flat_input.detach())
            
            # Commitment loss: ||z - sg[c_i*]||²
            commitment_loss = F.mse_loss(flat_input, c_i_star.detach())
            
            # Total loss
            loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return z_q, loss, indices
    
    def get_codebook_usage(self) -> float:
        """Get current codebook usage percentage"""
        if not hasattr(self, 'replacement'):
            return 0.0
        active = (self.replacement.usage_counter > 0).sum().item()
        return 100.0 * active / self.num_embeddings
    
    def compute_perplexity(self, indices: torch.Tensor) -> float:
        """
        Compute perplexity (average codebook usage) - Eq. 15
        Perplexity = exp(H(C)), where H(C) is entropy of codebook distribution
        """
        flat_indices = indices.flatten()
        unique, counts = torch.unique(flat_indices, return_counts=True)
        
        # Compute empirical probabilities
        probs = counts.float() / flat_indices.size(0)
        
        # Compute entropy H(C) = -Σ p_k * log(p_k)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Perplexity = exp(H(C))
        perplexity = torch.exp(entropy)
        
        return perplexity.item()


class DiVeQDetach(nn.Module):
    """
    DiVeQ-detach variant (Appendix B.2)
    Skips directional noise and uses stop gradient operator directly
    
    z_q = z + ||c_i* - z||₂ · sg((c_i* - z) / ||c_i* - z||₂)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        use_codebook_replacement: bool = True,
        replacement_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        self.use_codebook_replacement = use_codebook_replacement
        if use_codebook_replacement:
            config = replacement_config or {}
            self.replacement = CodebookReplacement(
                num_embeddings=num_embeddings,
                discard_threshold=config.get('discard_threshold', 0.01),
                replacement_interval=config.get('replacement_interval', 100),
            )
    
    def forward(
        self, 
        z: torch.Tensor,
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        input_shape = z.shape
        flat_input = z.reshape(-1, self.embedding_dim)
        
        # Find nearest codeword
        distances = torch.cdist(flat_input, self.codebook.weight)
        indices = torch.argmin(distances, dim=1)
        c_i_star = self.codebook.weight[indices]
        
        # Compute direction: c_i* - z
        d_vec = c_i_star - flat_input
        magnitude = torch.norm(d_vec, p=2, dim=1, keepdim=True)
        direction = (d_vec / (magnitude + 1e-8)).detach()  # Stop gradient
        
        # DiVeQ-detach: z_q = z + ||c_i* - z||₂ · sg((c_i* - z)/||c_i* - z||₂)
        z_q = flat_input + magnitude * direction
        
        z_q = z_q.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])
        
        if self.training and self.use_codebook_replacement:
            self.replacement.update_usage(indices.flatten())
            self.codebook.weight.data = self.replacement.replace_unused(
                self.codebook.weight.data
            )
        
        loss = None
        if return_loss:
            codebook_loss = F.mse_loss(c_i_star, flat_input.detach())
            commitment_loss = F.mse_loss(flat_input, c_i_star.detach())
            loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return z_q, loss, indices


class SFDiVeQ(nn.Module):
    """
    Space-Filling Differentiable Vector Quantization (SF-DiVeQ)
    
    SF-DiVeQ extends DiVeQ by quantizing along line segments connecting neighboring
    codewords, reducing quantization error and ensuring full codebook utilization.
    
    Based on: "DiVeQ: Differentiable Vector Quantization using the Reparameterization Trick"
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sigma_squared: float = 1e-2,
        commitment_cost: float = 0.25,
        init_warmup_epochs: int = 2,
    ):
        """
        Args:
            num_embeddings: Number of codewords in the codebook (K)
            embedding_dim: Dimension of each codeword (D)
            sigma_squared: Variance for directional noise (σ²)
            commitment_cost: Weight for commitment loss (β)
            init_warmup_epochs: Number of epochs to skip VQ for initialization
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sigma_squared = sigma_squared
        self.commitment_cost = commitment_cost
        self.init_warmup_epochs = init_warmup_epochs
        
        # Initialize codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        self.initialized = False
        self.latent_buffer = []  # Buffer for initialization
        
    def initialize_codebook(self, z: torch.Tensor, num_latents_per_code: int = 40):
        """
        Initialize codebook from recent latent vectors (Appendix A.5)
        Each codeword should be average of at least 20-40 recent latent vectors
        
        Args:
            z: Recent latent vectors [N, D]
            num_latents_per_code: Number of latents to average per codeword
        """
        with torch.no_grad():
            flat_z = z.reshape(-1, self.embedding_dim)
            n_latents = flat_z.size(0)
            
            # Need enough latents: K * num_latents_per_code
            required = self.num_embeddings * num_latents_per_code
            if n_latents < required:
                # Buffer latents until we have enough
                self.latent_buffer.append(flat_z)
                total_buffered = sum(buf.size(0) for buf in self.latent_buffer)
                if total_buffered < required:
                    return False
                # Concatenate buffered latents
                flat_z = torch.cat(self.latent_buffer, dim=0)
                self.latent_buffer.clear()
            
            # Initialize each codeword as average of recent latents
            indices = torch.randperm(flat_z.size(0))[:self.num_embeddings * num_latents_per_code]
            selected = flat_z[indices].reshape(self.num_embeddings, num_latents_per_code, -1)
            self.codebook.weight.data = selected.mean(dim=1)
            
        self.initialized = True
        return True
    
    def forward(
        self, 
        z: torch.Tensor,
        return_loss: bool = True,
        skip_quantization: bool = False,
        lambda_pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with SF-DiVeQ quantization
        
        Args:
            z: Input tensor of shape [B, D, H, W] or [B, L, D]
            return_loss: Whether to return the loss
            skip_quantization: Skip quantization during warmup (for initialization)
            
        Returns:
            z_q: Quantized output (differentiable) or original z if skipped
            loss: VQ loss (reconstruction + commitment)
            indices: Indices of selected codeword pairs
        """
        # During warmup, collect latents for initialization
        if skip_quantization or not self.initialized:
            if self.training and not self.initialized:
                # Try to initialize with current batch
                self.initialize_codebook(z)
            # Return input unchanged during warmup
            dummy_indices = torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device)
            return z, None, dummy_indices
        
        # Reshape input for computation
        input_shape = z.shape
        flat_input = z.reshape(-1, self.embedding_dim)  # [N, D]
        
        # Generate dithered codebook by interpolating consecutive codewords
        # Sample interpolation factors λ ~ U(0, 1)
        if lambda_pairs is not None:
            if lambda_pairs.dim() == 1:
                lambda_pairs = lambda_pairs.unsqueeze(-1)
            if lambda_pairs.shape[0] != self.num_embeddings - 1 or lambda_pairs.shape[1] != 1:
                raise ValueError("lambda_pairs has incompatible length.")
            lambda_vals = lambda_pairs.to(z.device, z.dtype)
        else:
            lambda_vals = torch.rand(self.num_embeddings - 1, 1, 
                                     device=z.device, dtype=z.dtype)  # [K-1, 1]
        
        # Create dithered codewords: c_d = (1-λ)·c_i + λ·c_{i+1}
        c_i = self.codebook.weight[:-1]  # [K-1, D]
        c_i_plus_1 = self.codebook.weight[1:]  # [K-1, D]
        dithered_codebook = (1 - lambda_vals) * c_i + lambda_vals * c_i_plus_1  # [K-1, D]
        
        # Find nearest dithered codeword
        distances = torch.cdist(flat_input, dithered_codebook)  # [N, K-1]
        flat_indices = torch.argmin(distances, dim=1)  # [N]
        
        # Get the two base codewords for each sample
        c_i_star = self.codebook.weight[flat_indices]  # [N, D]
        c_i_star_plus_1 = self.codebook.weight[flat_indices + 1]  # [N, D]
        lambda_i_star = lambda_vals[flat_indices].squeeze(-1)  # [N]
        
        # Compute directional vectors
        d_i = c_i_star - flat_input  # [N, D]
        d_i_plus_1 = c_i_star_plus_1 - flat_input  # [N, D]
        
        # Sample directional noise for both directions
        if self.training:
            v = torch.randn_like(d_i) * (self.sigma_squared ** 0.5)
            v_d_i = v + d_i
            v_d_i_plus_1 = v + d_i_plus_1
        else:
            # At inference, deterministic mapping
            v_d_i = d_i
            v_d_i_plus_1 = d_i_plus_1
        
        # Compute magnitudes
        magnitude_i = torch.norm(d_i, p=2, dim=1, keepdim=True)  # [N, 1]
        magnitude_i_plus_1 = torch.norm(d_i_plus_1, p=2, dim=1, keepdim=True)  # [N, 1]
        
        # Compute normalized directions with stop gradient
        direction_i = v_d_i / (torch.norm(v_d_i, p=2, dim=1, keepdim=True) + 1e-8)
        direction_i_plus_1 = v_d_i_plus_1 / (torch.norm(v_d_i_plus_1, p=2, dim=1, keepdim=True) + 1e-8)
        direction_i = direction_i.detach()
        direction_i_plus_1 = direction_i_plus_1.detach()
        
        # SF-DiVeQ quantization (Eq. 12)
        lambda_i_star = lambda_i_star.unsqueeze(-1)  # [N, 1]
        z_q = (flat_input + 
               magnitude_i * (1 - lambda_i_star) * direction_i + 
               magnitude_i_plus_1 * lambda_i_star * direction_i_plus_1)
        
        # Reshape back to original shape
        z_q = z_q.reshape(input_shape)
        indices = flat_indices.reshape(input_shape[:-1])
        
        # Compute loss
        loss = None
        if return_loss:
            # For SF-DiVeQ, compute loss w.r.t. dithered codewords
            dithered_targets = dithered_codebook[flat_indices]
            
            # Codebook loss
            codebook_loss = F.mse_loss(dithered_targets, flat_input.detach())
            
            # Commitment loss
            commitment_loss = F.mse_loss(flat_input, dithered_targets.detach())
            
            # Total loss
            loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return z_q, loss, indices
    
    def get_codebook_usage(self) -> float:
        """
        Get codebook usage percentage
        SF-DiVeQ uses all codewords by design (quantizes on line segments)
        """
        if not self.initialized:
            return 0.0
        return 100.0  # SF-DiVeQ achieves full utilization
    
    def compute_perplexity(self, indices: torch.Tensor) -> float:
        """Compute perplexity for SF-DiVeQ"""
        flat_indices = indices.flatten()
        unique, counts = torch.unique(flat_indices, return_counts=True)
        probs = counts.float() / flat_indices.size(0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return torch.exp(entropy).item()


class SFDiVeQDetach(nn.Module):
    """
    SF-DiVeQ-detach variant (Appendix B.2)
    Skips directional noise and uses stop gradient directly
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        init_warmup_epochs: int = 2,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.init_warmup_epochs = init_warmup_epochs
        
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        self.initialized = False
        self.latent_buffer = []
    
    def initialize_codebook(self, z: torch.Tensor, num_latents_per_code: int = 40):
        """Initialize codebook from recent latents"""
        with torch.no_grad():
            flat_z = z.reshape(-1, self.embedding_dim)
            n_latents = flat_z.size(0)
            required = self.num_embeddings * num_latents_per_code
            
            if n_latents < required:
                self.latent_buffer.append(flat_z)
                total_buffered = sum(buf.size(0) for buf in self.latent_buffer)
                if total_buffered < required:
                    return False
                flat_z = torch.cat(self.latent_buffer, dim=0)
                self.latent_buffer.clear()
            
            indices = torch.randperm(flat_z.size(0))[:self.num_embeddings * num_latents_per_code]
            selected = flat_z[indices].reshape(self.num_embeddings, num_latents_per_code, -1)
            self.codebook.weight.data = selected.mean(dim=1)
            
        self.initialized = True
        return True
    
    def forward(
        self, 
        z: torch.Tensor,
        return_loss: bool = True,
        skip_quantization: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if skip_quantization or not self.initialized:
            if self.training and not self.initialized:
                self.initialize_codebook(z)
            dummy_indices = torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device)
            return z, None, dummy_indices
        
        input_shape = z.shape
        flat_input = z.reshape(-1, self.embedding_dim)
        
        # Generate dithered codebook
        lambda_vals = torch.rand(self.num_embeddings - 1, 1, 
                                 device=z.device, dtype=z.dtype)
        c_i = self.codebook.weight[:-1]
        c_i_plus_1 = self.codebook.weight[1:]
        dithered_codebook = (1 - lambda_vals) * c_i + lambda_vals * c_i_plus_1
        
        distances = torch.cdist(flat_input, dithered_codebook)
        indices = torch.argmin(distances, dim=1)
        
        c_i_star = self.codebook.weight[indices]
        c_i_star_plus_1 = self.codebook.weight[indices + 1]
        lambda_i_star = lambda_vals[indices].squeeze(-1)
        
        # Directional vectors (no noise)
        d_i = c_i_star - flat_input
        d_i_plus_1 = c_i_star_plus_1 - flat_input
        
        magnitude_i = torch.norm(d_i, p=2, dim=1, keepdim=True)
        magnitude_i_plus_1 = torch.norm(d_i_plus_1, p=2, dim=1, keepdim=True)
        
        direction_i = (d_i / (magnitude_i + 1e-8)).detach()
        direction_i_plus_1 = (d_i_plus_1 / (magnitude_i_plus_1 + 1e-8)).detach()
        
        lambda_i_star = lambda_i_star.unsqueeze(-1)
        z_q = (flat_input + 
               magnitude_i * (1 - lambda_i_star) * direction_i + 
               magnitude_i_plus_1 * lambda_i_star * direction_i_plus_1)
        
        z_q = z_q.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])
        
        loss = None
        if return_loss:
            dithered_targets = dithered_codebook[indices]
            codebook_loss = F.mse_loss(dithered_targets, flat_input.detach())
            commitment_loss = F.mse_loss(flat_input, dithered_targets.detach())
            loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return z_q, loss, indices


# Example usage
# if __name__ == "__main__":
#     # Parameters
#     batch_size = 4
#     channels = 256
#     height = 16
#     width = 16
#     num_embeddings = 512
#     embedding_dim = 256
    
#     # Create input
#     z = torch.randn(batch_size, embedding_dim, height, width)
    
#     print("=== DiVeQ Example ===")
#     diveq = DiVeQ(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#     diveq.train()
#     z_q, loss, indices = diveq(z)
#     print(f"Input shape: {z.shape}")
#     print(f"Output shape: {z_q.shape}")
#     print(f"Loss: {loss.item():.4f}")
#     print(f"Indices shape: {indices.shape}")
#     print(f"Unique codewords used: {len(torch.unique(indices))}/{num_embeddings}")
#     print(f"Codebook usage: {diveq.get_codebook_usage():.1f}%")
#     print(f"Perplexity: {diveq.compute_perplexity(indices):.2f}")
    
#     print("\n=== DiVeQ-detach Example ===")
#     diveq_detach = DiVeQDetach(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#     diveq_detach.train()
#     z_q, loss, indices = diveq_detach(z)
#     print(f"Output shape: {z_q.shape}")
#     print(f"Loss: {loss.item():.4f}")
#     print(f"Unique codewords used: {len(torch.unique(indices))}/{num_embeddings}")
    
#     print("\n=== SF-DiVeQ Example ===")
#     sf_diveq = SFDiVeQ(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#     sf_diveq.train()
    
#     # Simulate warmup initialization
#     print("Initializing SF-DiVeQ...")
#     for i in range(3):
#         z_warmup = torch.randn(batch_size, embedding_dim, height, width)
#         z_q, loss, indices = sf_diveq(z_warmup, skip_quantization=True)
#         if sf_diveq.initialized:
#             print(f"Initialized after {i+1} batches")
#             break
    
#     # Now quantize
#     z_q, loss, indices = sf_diveq(z)
#     print(f"Output shape: {z_q.shape}")
#     print(f"Loss: {loss.item():.4f}")
#     print(f"Unique codeword pairs used: {len(torch.unique(indices))}/{num_embeddings-1}")
#     print(f"Codebook usage: {sf_diveq.get_codebook_usage():.1f}% (always 100% by design)")
#     print(f"Perplexity: {sf_diveq.compute_perplexity(indices):.2f}")
    
#     print("\n=== SF-DiVeQ-detach Example ===")
#     sf_diveq_detach = SFDiVeQDetach(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#     sf_diveq_detach.train()
    
#     # Initialize
#     for i in range(3):
#         z_warmup = torch.randn(batch_size, embedding_dim, height, width)
#         z_q, loss, indices = sf_diveq_detach(z_warmup, skip_quantization=True)
#         if sf_diveq_detach.initialized:
#             break
    
#     z_q, loss, indices = sf_diveq_detach(z)
#     print(f"Output shape: {z_q.shape}")
#     print(f"Loss: {loss.item():.4f}")
    
#     print("\n=== Comparison Summary ===")
#     print("DiVeQ: Standard differentiable VQ with σ² = 1e-3")
#     print("DiVeQ-detach: Variant without directional noise")
#     print("SF-DiVeQ: Space-filling variant with σ² = 1e-2, no codebook replacement needed")
#     print("SF-DiVeQ-detach: Space-filling variant without directional noise")
