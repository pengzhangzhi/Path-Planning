import torch
from typing import Callable, Tuple


def logP(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Compute the log probabilities of predicted tokens from model logits.
    
    This function calculates the log probability of each token in x0 according to the model's
    predicted distribution. It's a common scoring function used in P2 sampling to determine
    the confidence of the model in each position.
    
    Mathematical formulation:
    logP(x_0) = log(p(x_0 | logits)) = log_softmax(logits)[x_0]
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) - Raw logits from model
        x0: Tensor of shape (batch_size, seq_len) - Token indices to compute probabilities for
        
    Returns:
        scores: Tensor of shape (batch_size, seq_len) - Log probabilities for each token
    """
    logits = logits.double()
    logits = logits.log_softmax(dim=-1)
    scores = logits.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    return scores


def random_score(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Return a random score for each token.
    
    This function generates random scores irrespective of the logits or token values.
    It's used for pure diffusion sampling where token selection is random rather than
    based on model confidence.
    
    Mathematical formulation:
    score(x) = log(rand(0, 1))
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) - Raw logits from model (unused)
        x0: Tensor of shape (batch_size, seq_len) - Token indices (used only for shape)
        
    Returns:
        Tensor of shape (batch_size, seq_len) - Random log scores for each position
    """
    return torch.rand_like(x0.float()).log()


def diff_top2(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Compute the difference between the top 2 probabilities for each position.
    
    This scoring function measures model confidence as the gap between the most likely
    and second most likely tokens at each position. A larger difference indicates higher
    confidence.
    
    Mathematical formulation:
    score(x) = log_softmax(logits)[top_1] - log_softmax(logits)[top_2]
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) - Raw logits from model
        x0: Tensor of shape (batch_size, seq_len) - Token indices (unused in this function)
        
    Returns:
        Tensor of shape (batch_size, seq_len) - Difference between top 2 log probabilities
    """
    logits = logits.log_softmax(dim=-1)
    top2_logits = logits.topk(2, dim=-1).values
    diff = top2_logits[:, :, 0] - top2_logits[:, :, 1]
    return diff



