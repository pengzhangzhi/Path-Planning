import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from path_planning.p2 import *
from path_planning.utils import seed_everything
from LLaDA_hf.modeling_llada import LLaDAModelLM

class ModelWrapper:
    """Wrapper for the model to handle logits processing."""
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x):
        outputs = self.model(x)
        return outputs.logits


def main():
    device = 'cuda'
    seed_everything(42)  # For reproducibility

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    model_wrapper = ModelWrapper(model)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    gen_length = 128
    mask_id = 126336
    xt = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    xt[:, :input_ids.shape[1]] = input_ids.clone()
    
    
    print(f"Input shape: {input_ids.shape}, Full sequence shape: {xt.shape}")
    
    
    sampled_xt = p2_plus_sampling(
        xt=xt,
        model=model_wrapper,
        mask_id=mask_id,
        num_steps=128,
        tau=0.,
    )
    print(f'prompt: {prompt}')
    print(f'generated: {tokenizer.batch_decode(sampled_xt[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}')


if __name__ == '__main__':
    main()