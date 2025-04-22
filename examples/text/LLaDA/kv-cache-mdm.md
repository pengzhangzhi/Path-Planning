# Implementation Plan: KVCache for Masked Diffusion Model (MDM)

This document outlines a clear, structured, and detailed implementation plan to introduce a **Key-Value (KV) Cache** mechanism into a **Masked Diffusion Model (MDM)**, using a **bidirectional Transformer** architecture similar to BERT with **RoPE (Rotary Positional Encoding)**.

---

## 📌 Overview

The KVCache mechanism consists of two distinct stages:

### 1. Prefilling Stage

**Input**: A full input sequence `X_T` containing both masked and unmasked tokens.

**Goals**:
- Compute and store Key (`K`) and Value (`V`) tensors for all tokens.
- Compute logits for all masked tokens to determine their **decoding order**.

### 2. Decoding Stage

**Input**:
- Original input sequence `X_T`.
- KVCache computed during prefill.
- A list of token positions to decode (`decode_indices`).

**Goals**:
- Compute the Query (`Q`) for the selected masked positions.
- Perform attention using `Q` and cached `K`, `V`.
- Predict and unmask the selected tokens.
- Update the cached `K`, `V` at the decoded positions.

---

## 🧠 Detailed Implementation Plan

### A. Model Forward Function

Extend the model’s `forward()` function to support an additional argument:

```python
def forward(
    self,
    input_ids,
    decode_indices: Optional[List[int]] = None,
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
    use_cache: bool = False,
):
    ...
```
•	If decode_indices is None, perform the prefilling step.
•	If decode_indices is provided, perform decoding only for the specified masked token positions.

B. Prefilling Stage

Steps:
	1.	Run a full forward pass on the entire input sequence X_T.
	2.	In each attention layer:
	•	Compute Keys (K) and Values (V) for all tokens.
	•	Store (K, V) in a cache (KVCache).
	3.	Compute logits for all masked tokens.
	4.	Rank masked positions based on prediction scores to determine decoding order.

KVCache Format:

```python
KVCache = [
    (K_layer_1, V_layer_1),
    (K_layer_2, V_layer_2),
    # ...
]
# Each K, V shape: [batch_size, num_heads, seq_len, head_dim]
```
Prefill Output:
•	KVCache: Cached K and V for all positions and layers.
•	decoding_order: List of masked indices sorted by predicted confidence.


C. Decoding Stage

Inputs:
	•	X_T: Input sequence with remaining masked tokens.
	•	decode_indices: List of token indices to decode in this step.
	•	KVCache: Cached keys and values from the prefill stage.

Steps:
	1.	For each decode_index in decode_indices, extract current token embedding.
	2.	Compute Q only for the selected masked positions.
	3.	Use cached K, V for attention (with option to update K/V at decoded indices).
	4.	Perform attention:
    ```python
    attn_output = Attention(Q, K, V)
    ```
    5.	Pass attn_output through the MLP only at decode_indices.
	6.	Replace mask tokens at decode_indices with predicted tokens.
	7.	Update KVCache at decode_indices using new token embeddings.

Tensor Shapes:
	•	Q: [batch_size, num_heads, num_to_decode, head_dim]
	•	K, V: [batch_size, num_heads, seq_len, head_dim]


D. RoPE Positional Encoding

Problem:
	•	Existing RoPE assumes left-to-right incremental positions.
	•	MDM uses arbitrary positions for decoding, so RoPE must encode explicit indices.

Solution:

Implement a function to encode RoPE with arbitrary position indices:
```python
def rope_encode_query(query_embedding, position_index):
    """
    Applies RoPE to query embeddings using the explicit token position.
    """
    return apply_rope(query_embedding, position=position_index)
```


🧪 Example Usage Workflow (Pseudo-code)

```python
# --- Prefill Stage ---
outputs = model(input_ids=X_T, use_cache=True)
KVCache = outputs.past_key_values
logits = outputs.logits
masked_positions = get_masked_positions(X_T)
decoding_order = sort_masked_positions_by_logits(logits, masked_positions)

# --- Decoding Stage ---
for decode_indices in chunked(decoding_order, batch_size):
    outputs = model(
        input_ids=X_T,
        decode_indices=decode_indices,
        past_key_values=KVCache,
        use_cache=True
    )
    predicted_tokens = outputs.logits.argmax(-1)
    X_T[decode_indices] = predicted_tokens
    KVCache = outputs.past_key_values
```

🧩 Integration with Codebase
	•	Clearly document all new functions (rope_encode_query, decoding logic).
	•	Update forward() API consistently across model subclasses.
	•	Ensure compatibility with any internal modules consuming model output.
	•	If breaking changes occur (e.g., forward signature), clearly version or isolate modified class.

⸻

✅ Implementation Notes & Corner Cases
	•	Attention must remain fully bidirectional (no causal mask).
	•	KVCache must be indexed and updated per-layer and per-position.
	•	Positional encoding must remain stable and position-dependent, not step-dependent.
	•	All decode operations must be batch-parallelizable over decode_indices.

⸻

🔍 Suggested Unit Tests
	1.	RoPE Encoding: Validate outputs for random position indices.
	2.	KVCache Shape: Confirm cache structure after prefill.
	3.	Decoding Output: Ensure single-step and multi-token decoding is correct.
	4.	Cache Update: Ensure cache is correctly updated only at specified decode positions.
	5.	Full Decoding: Ensure final decoded sequence matches expected output when run sequentially.

⸻

📦 Final Prompt for AI-Assisted Implementation

“Implement KVCache support for a masked diffusion model using a BERT-style bidirectional transformer. The model’s forward pass supports two modes: prefill (full sequence) and decode (specific masked token indices). In prefill mode, compute and cache Keys and Values for all tokens. In decode mode, select the token indices to decode, compute queries for those indices, perform attention using cached K/V, and update outputs and cache. Use rotary positional encoding (RoPE), and ensure positional encoding for queries is computed using explicit token indices. Support batch decoding for multiple masked positions in one step.”


