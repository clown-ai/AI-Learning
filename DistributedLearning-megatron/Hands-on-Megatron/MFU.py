import math


"""
    Notice that, when calculate Matrix Multiply for A_{m x k} X B_{k x n},
    there is 2 x m x k x n FLOPs.
    Why there is 2 because we not only hava mul but also have add op in a calcualtion
"""
class MFU:
    def __init__(self, batch_size, seq_length, hidden_size, ffn_size, num_layers, 
                 vocab_size, model_arch, attn_mode, num_gqa=0, attn_heads=0):
        self.B = batch_size
        self.s = seq_length
        self.h = hidden_size
        self.f = ffn_size
        self.L = num_layers
        self.V = vocab_size
        self.num_gqa = num_gqa
        self.attn_heads = attn_heads
        self.model_arch = model_arch
        self.attn_mode = attn_mode

    def calculate_attn(self):
        if self.attn_mode == "MHA":
            # Q, K, V projection: [s, b, h] * [h, h] ==> [s, b, h]
            self.QKV_proj = 2 * self.s * self.B * math.pow(self.h, 2) * 3
            # Q * K_T: [s, b, h], [s, b, h] --> [b, s, h] * [b, h, s] ==> [b, s, s]
            self.QK_T = 2 * self.B * math.pow(self.s, 2) * self.h
            # attn_score * V: [b, s, s] * [b, s, h] ==> [b, s, h]
            self.score_V = 2 * self.B * math.pow(self.s, 2) * self.h
            # linear_projection for attn output: [b, s, h] * [h, h] ==> [b, s, h]
            self.O_proj = 2 * self.B * math.pow(self.h, 2) * self. s
        if self.attn_mode == "GQA":
             # Q, K, V projection: [s, b, h] * [h, h] ==> [s, b, h]
            self.QKV_proj = 2 * self.s * self.B * math.pow(self.h, 2) + 4 * self.s * self.B * math.pow(self.h, 2) * self.num_gqa / self.attn_heads
            # Q * K_T: [s, b, h], [s, b, h] --> [b, s, h] * [b, h, s] ==> [b, s, s]
            self.QK_T = 2 * self.B * math.pow(self.s, 2) * self.h
            # attn_score * V: [b, s, s] * [b, s, h] ==> [b, s, h]
            self.score_V = 2 * self.B * math.pow(self.s, 2) * self.h
            # linear_projection for attn output: [b, s, h] * [h, h] ==> [b, s, h]
            self.O_proj = 2 * self.B * math.pow(self.h, 2) * self. s

        return self.QKV_proj + self.QK_T + self.score_V + self.O_proj

    
    def calculate_mlp(self):
        if self.model_arch == "gpt":
            # up projection: [s, b, h] * [h, f]
            self.up = 2 * self.s * self.B * self.h * self.f
            # down projection: [s, b, f] * [f, h]
            self.down = 2 * self.s * self.B * self.f * self.h

            return self.up + self.down
        
        if self.model_arch == "llama":
            # up porjection: [s, b, h] * [h, f]:
            self.up = 2 * self.s * self.B * self.h * self.f
            # gate projection: [s, b, h] * [h, f]
            self.gate = 2 * self.s * self.B * self.h * self.f
            # down projection: [s, b, f] * [f, h]
            self.down = 2 * self.s * self.B * self.f * self.h

            return self.up + self.down + self.gate
        

    def summing_up(self):
        # vocab projection: [s, b, h] * [h, V]
        self.vocab_proj = 2 * self.s * self.B * self.h * self.V
        self.attn = self.calculate_attn()
        self.mlp = self.calculate_mlp()

        # summing up above is a forward pass, we still have backward pass(2 * forward pass)
        return (self.vocab_proj + self.L * (self.attn + self.mlp)) * 3
    

    def p_class(self):
        total_flops = self.summing_up()
        p_class_flops = total_flops / 1e+15
        return p_class_flops
    
    
    def t_flops_per_gpu(self, gpus, iter_time):
        total_flops = self.summing_up() / 1e+12
        return total_flops / gpus / iter_time
    

    def mfu(self, peak_flops, gpus, iter_time):
        per_gpu_flops = self.t_flops_per_gpu(gpus, iter_time)
        return per_gpu_flops / peak_flops


gpt_mfu = MFU(1024, 2048, 12288, 4*12288, 96, 51200, "gpt", "MHA")
GPT_MFU = gpt_mfu.mfu(313, 128, 107)
print("GPT3-175B MFU: {:.3f}%".format(GPT_MFU * 100))

llama_mfu = MFU(256, 4096, 4096, 11008, 32, 32000, "llama", "MHA")
LLAMA_MFU = llama_mfu.mfu(353, 8, 30.9)
print("LLAMA2-7B MFU: {:.3f}%".format(LLAMA_MFU * 100))

llama_mfu = MFU(512, 4096, 5120, 13824, 40, 32000, "llama", "MHA")
LLAMA_MFU = llama_mfu.mfu(353, 8, 131.71)
print("LLAMA2-13B MFU: {:.3f}%".format(LLAMA_MFU * 100))

llama_mfu = MFU(1024, 4096, 8192, 28672, 80, 32000, "llama", "GQA", 8, 64)
LLAMA_MFU = llama_mfu.mfu(353, 32, 306.5)
print("LLAMA2-70B MFU: {:.3f}%".format(LLAMA_MFU * 100))

# GPT3-175B MFU: 52.660%
# LLAMA2-7B MFU: 55.378%
# LLAMA2-13B MFU: 49.152%
# LLAMA2-13B MFU: 53.9%
