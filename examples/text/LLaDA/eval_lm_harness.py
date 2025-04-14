'''
This file is inspired by the code from 
- https://github.com/ML-GSAI/SMDM
- https://github.com/ML-GSAI/LLaDA/blob/main/evaluation/eval_llada.py
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from path_planning.p2 import p2_sampling

from transformers import AutoTokenizer, AutoModel


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        device="cuda",
        max_length=None,
        num_steps=None,
        tau=1.0,
        eta=1.0,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
            num_steps: The number of steps for the diffusion process.
            tau: The temperature for the diffusion process.
            eta: The eta for the diffusion process.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.is_instruct_ft = 'Instruct' in model_path
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.num_steps = num_steps
        self.tau = tau
        self.cfg = cfg
        self.eta = eta
        print(f'model: {model_path}')
        print(f'Is check greedy: {is_check_greedy}')
        print(f'cfg: {cfg}')
        print(f'num_steps: {num_steps}')
        print(f'tau: {tau}')
        print(f'eta: {eta}')
        if self.accelerator is not None:
            print(f'Running with accelerate on {self.accelerator.num_processes} processes')
            print(f'Local process index: {self._rank}')
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
                print('=' * 20)
                print('prefix: ', elem['prefix_text'])
                print('target: ', elem['target_text'])
                print(ll, is_target_greedy_dec)
                print('=' * 20, end='\n\n')
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    def generate_until(self, requests,):
        """
        Generates text continuations for a list of requests using the masked diffusion approach,
        processing requests in batches of size `small_batch_size`.

        Each request should have:
        - args[0]: A prompt string.
        - args[1]: A dictionary of generation parameters (e.g., {"max_gen_toks": 128, "until": ["stopword1", ...]}).

        Returns:
        A list of generated text strings.
        """
        batch_size = self.batch_size
        outputs = []
        
        # Process requests in batches of small_batch_size
        for batch_start in range(0, len(requests), batch_size):
            batch_requests = requests[batch_start : batch_start + batch_size]
            
            prompts = []
            stop_words_list = []
            max_gen_toks_list = []

            # Process each request in the current batch to extract prompt and per-request parameters
            for req in batch_requests:
                prompt_text = req.args[0]
                gen_params = req.args[1] if len(req.args) > 1 else {}
                max_gen_toks = self.max_length if self.max_length is not None else gen_params.get("max_gen_toks", self.max_length)
                stop_words = gen_params.get("until", [])
                
                # For instruct finetuned models, apply the chat template
                if self.is_instruct_ft:
                    m = [{"role": "user", "content": prompt_text}]
                    prompt_text = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                
                prompts.append(prompt_text)
                stop_words_list.append(stop_words)
                max_gen_toks_list.append(max_gen_toks)
            
            # Tokenize all prompts together; use padding to create a uniform tensor shape
            tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)  # shape: (batch_size, max_prompt_length)
            batch_size, max_prompt_length = input_ids.shape
            prompt_lengths = tokenized["attention_mask"].sum(dim=1)  # shape: (batch_size,)
            
            # Determine the maximum generation tokens for this batch
            max_gen_toks_batch = max(max_gen_toks_list)
            
            # Create a batched tensor filled with mask_id with room for prompt and generated tokens
            xt = torch.full((batch_size, max_prompt_length + max_gen_toks_batch),
                            self.mask_id, dtype=torch.long, device=self.device)
            
            # Copy each prompt into its corresponding row in the batched tensor
            for i in range(batch_size):
                prompt_len = prompt_lengths[i].item()
                xt[i, :prompt_len] = input_ids[i, :prompt_len]
            
            # Use a common number of steps (using self.num_steps if set, else use max_gen_toks_batch)
            num_steps = self.num_steps if self.num_steps is not None else max_gen_toks_batch
            tau = self.tau
            eta = self.eta

            # Define a batched model wrapper that returns logits for the entire batch
            def model_wrapper(x):
                outputs = self.model(x)
                return outputs.logits

            # Run the diffusion-based sampling on the current batch
            sampled_xt = p2_sampling(
                xt=xt,
                model=model_wrapper,
                mask_id=self.mask_id,
                num_steps=num_steps,
                tau=tau,
                eta=eta,
            )
            
            # Process each sample in the current batch: extract generated tokens and decode them
            for i in range(batch_size):
                prompt_len = prompt_lengths[i].item()
                gen_limit = max_gen_toks_list[i]
                gen_tokens = sampled_xt[i, prompt_len:prompt_len + gen_limit]
                generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                # Truncate generated text by any specified stop words
                for word in stop_words_list[i]:
                    if word in generated_text:
                        generated_text = generated_text[:generated_text.index(word)]
                
                print(generated_text)
                print('=' * 20, end='\n\n')
                outputs.append(generated_text)
        
        return outputs
    

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    