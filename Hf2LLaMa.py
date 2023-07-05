import torch
import sys

# Can be in a separate file - params.json
params = { "dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": -1 }
# Can be used to pass as an arg
quantized = True

def dict_list(d, prefix=''):
    for k, v in d.items():
        if k.startswith(prefix):
            print(k, v.shape)

def unpermute(w):
    n_heads = params['n_heads']
    dim = params['dim'] 
    return w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)

def load_qweights_and_unquantize(loaded, prefix):
    qw = loaded[prefix + ".qweight"]
    qs = loaded[prefix + ".scales"]
    qz = loaded[prefix + ".zeros"]
    cols = qw.shape[0] * 8
    fb = (qw.repeat_interleave(8, dim=0) >> (torch.arange(cols).reshape(cols, 1) % 8 * 4)) & 15
    return (fb.t() * qs - qz).half()

def load_weights(loaded, prefix):
    return loaded[prefix + ".weight"]
    
def unhug_model(loaded, load_fn=load_weights):
    out = {}
    for layer_i in range(32):
        part = {
            f"layers.{layer_i}.attention.wq.weight": unpermute(load_fn(loaded, f"model.layers.{layer_i}.self_attn.q_proj")),
            f"layers.{layer_i}.attention.wk.weight": unpermute(load_fn(loaded, f"model.layers.{layer_i}.self_attn.k_proj")),
            f"layers.{layer_i}.attention.wv.weight": load_fn(loaded, f"model.layers.{layer_i}.self_attn.v_proj"),
            f"layers.{layer_i}.attention.wo.weight": load_fn(loaded, f"model.layers.{layer_i}.self_attn.o_proj"),
            f"layers.{layer_i}.feed_forward.w1.weight": load_fn(loaded, f"model.layers.{layer_i}.mlp.gate_proj"),
            f"layers.{layer_i}.feed_forward.w2.weight": load_fn(loaded, f"model.layers.{layer_i}.mlp.down_proj"),
            f"layers.{layer_i}.feed_forward.w3.weight": load_fn(loaded, f"model.layers.{layer_i}.mlp.up_proj"),
            f"layers.{layer_i}.attention_norm.weight": loaded[f"model.layers.{layer_i}.input_layernorm.weight"],
            f"layers.{layer_i}.ffn_norm.weight": loaded[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
        }
        out.update(part)
        out["tok_embeddings.weight"] = loaded["model.embed_tokens.weight"]
        out["norm.weight"] = loaded["model.norm.weight"]
        out["output.weight"] = loaded["lm_head.weight"]

    return out    


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: convert.py [llama-4bit-quant-in-hf-format] [output-fp16-llama-format]')
        sys.exit(1)
    
    if True:
        hf_model = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    else:
        hf_model = {}
        for i in range(33):
            part = torch.load(f"/models/llama-7b-hf/pytorch_model-000{i+1:02}-of-00033.bin", map_location="cpu", weights_only=True)
            hf_model.update(part)

    ll_model = unhug_model(hf_model, load_qweights_and_unquantize if quantized else load_weights)
    torch.save(ll_model, sys.argv[2])
