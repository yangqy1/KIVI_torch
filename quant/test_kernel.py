import torch
import numpy as np
from new_pack import unpack_and_dequant_vcache
from quant.matmul import cuda_bmm_fA_qB_outer, triton_bmm_fA_qB_outer
def simulate_bit_packing(weights, bit):
    # Simulating packed weight behavior
    pack_factor = 16  # as per your CUDA code
    OC, IC = weights.shape
    packed_weights = torch.zeros((OC // pack_factor, IC), dtype=torch.float32)
    for idx in range(0, OC, pack_factor):
        # This is a very simplified simulation of bit packing
        packed_weights[idx // pack_factor, :] = weights[idx:idx + pack_factor, :].sum(0)
    return packed_weights

def gemv_forward_pytorch(_in_feats, _kernel, _scaling_factors, _zeros, bit, group_size, nh, mqa):
    BS = _in_feats.size(0)
    num_in_feats = _in_feats.size(1)
    num_in_channels = _in_feats.size(2)
    num_out_channels = _zeros.size(1) * group_size
    print('ker',_kernel)
    # Convert data types to float16 and int32 as per the original data pointers
    in_feats = _in_feats.to(dtype=torch.float16)
    kernel = _kernel.to(dtype=torch.int32)
    zeros = _zeros.to(dtype=torch.float16)
    scaling_factors = _scaling_factors.to(dtype=torch.float16)

    # Configure tensor options for output features
    options = {"dtype": _in_feats.dtype, "device": _in_feats.device}
    _out_feats = torch.empty((BS, num_in_feats, num_out_channels), **options)
    num_out_feats = _out_feats.size(-2)
    out_feats = _out_feats.to(dtype=torch.float16)

    # Calculate packing factor based on bit size
    pack_factor = 32 // bit
    num_blocks = (BS, (num_out_channels // pack_factor + 3) // 4, num_out_feats)
    num_threads = (32, 4)
    # print(kernel)
    print('Input tensors:')
    print_tensor_info('_in_feats', _in_feats)
    print_tensor_info('_kernel', _kernel)
    print_tensor_info('_scaling_factors', _scaling_factors)
    print_tensor_info('_zeros', _zeros)

    print('Constants:')
    print(f'num_in_channels: {num_in_channels}')
    print(f'num_out_channels: {num_out_channels}')
    print(f'group_size: {group_size}')
    print(f'nh: {nh}')
    print(f'mqa: {mqa}')
    # Launch custom CUDA kernels
    if bit == 4:
        # Custom kernel function for bit == 4, needs to be implemented in PyTorch CUDA extensions
        bgemv4_kernel_outer_dim(in_feats, kernel, zeros, scaling_factors, out_feats,
                                num_in_channels, num_out_channels, group_size, nh, mqa)
    else:
        # Custom kernel function for other bit sizes, specifically 2 here, also needs CUDA extensions
        # bgemv2_forward_pytorch(in_feats, kernel, zeros, scaling_factors, out_feats,
                                # num_in_channels, num_out_channels, group_size, nh, mqa)
        bgemv2_pytorch(in_feats, kernel, zeros, scaling_factors,  nh, mqa, group_size ) 
                   
# def bgemv2_forward_pytorch(in_feats, kernel, zeros, scaling_factors, bit, group_size, nh, mqa):
    return _out_feats

def print_tensor_info(name, tensor):
    """Print the shape and first few values of a tensor."""
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} values (first 10 elements): {tensor.flatten()[:10]}")

def bgemv2_kernel_outer_dim_pytorch(_inputs, _weight, _zeros, _scale, _outputs, IC, OC, group_size, nh, mqa):
    bit = 2
    pack_factor = 16
    num_batches = _inputs.shape[0]
    num = 0xFF >> (8 - bit)
    print(_outputs.shape)
    # Reshape and prepare tensors
    inputs = _inputs.view(num_batches, -1)  # Flatten inputs to [BS, IC]
    outputs = _outputs.view(num_batches, -1)  # Flatten outputs to [BS, OC]
    print(outputs.shape)
    for batch_idx in range(num_batches):
        # Adjust batch index for mqa (multi-query adjustment)
        if mqa:
            _batch_idx = batch_idx // nh
        else:
            _batch_idx = batch_idx
        print(_weight)
        weight = _weight[_batch_idx * OC * IC // pack_factor]
        scaling_factors = _scale[_batch_idx * OC * IC // group_size]
        zeros = _zeros[_batch_idx * OC * IC // group_size]

        # Iterate over the output channels in pack_factor groups
        for packed_oc_idx in range(0, OC, pack_factor):
            oc_start_idx = packed_oc_idx
            group_idx = oc_start_idx // group_size

            # Placeholder for packed weight processing
            for ic_idx in range(IC):
                # Simulate unpacking and processing of quantized weights
                cur_input = inputs[batch_idx, ic_idx]
                cur_scale = scaling_factors[packed_oc_idx % group_size]
                cur_zero = zeros[packed_oc_idx % group_size]
                
                for ic_pack in range(pack_factor):
                    oc_idx = oc_start_idx + ic_pack
                    if oc_idx < OC:
                        # weight_index = (_batch_idx * OC * IC // pack_factor) + packed_oc_idx * IC + k * TILE_DIM + threadIdx.x * 4
                        cur_weight = (weight[oc_idx] & num).float()  # Simplified weight access
                        print((cur_scale ).shape,  cur_weight.shape, cur_zero.shape)
                        dequantized_weight = cur_scale * cur_weight + cur_zero
                        
                        print((cur_scale ),  cur_weight, cur_zero)
                        print(weight, batch_idx, oc_idx, outputs.shape)
                        outputs[batch_idx, oc_idx] += dequantized_weight * cur_input
    # Reshape outputs back to the original shape
    _outputs[:] = outputs.view_as(_outputs)


def bgemv2_pytorch(inputs, weight, zeros, scale, nh, mqa, group_size):
    IC = inputs.size(1)
    OC = weight.size(0)
    batch_size = inputs.size(0)
    if mqa:
        batch_size //= nh

    outputs = torch.zeros((batch_size, OC), dtype=torch.float16, device=inputs.device)

    TILE_DIM = 128
    pack_factor = 16
    num_tiles = (IC + TILE_DIM - 1) // TILE_DIM
    
    for batch_idx in range(batch_size):
        if mqa:
            _batch_idx = batch_idx // nh
        else:
            _batch_idx = batch_idx

        # Slice weights, scaling factors, and zeros for current batch
        current_weight = weight[_batch_idx]
        current_scaling_factors = scale[_batch_idx]
        current_zeros = zeros[_batch_idx]
        input_batch = inputs[batch_idx]

        psum = torch.zeros(pack_factor, dtype=torch.float32, device=inputs.device)
        
        for k in range(num_tiles):
            weight_offset = k * TILE_DIM
            scale_mn_offset = (weight_offset // group_size) * IC
            inputs_ptr_delta = weight_offset
            
            inp = input_batch[inputs_ptr_delta:inputs_ptr_delta + TILE_DIM]
            qw = current_weight[weight_offset:weight_offset + TILE_DIM]
            cscale = current_scaling_factors[scale_mn_offset:scale_mn_offset + TILE_DIM]
            czero = current_zeros[scale_mn_offset:scale_mn_offset + TILE_DIM]

            for ic_0 in range(4):
                cur_packed_weight = qw[ic_0]
                print(f"cur_packed_weight: {cur_packed_weight.shape}")
                cur_inp = inp[ic_0].float()
                cur_scale = cscale[ic_0].float()
                cur_zero = czero[ic_0].float()

                # Unpack the packed weight and apply scaling
                for ic_1 in range(pack_factor):
                    oc_idx = ic_1
                    if oc_idx < OC:
                        cur_single_weight_fp = (cur_packed_weight & (0xFF >> (8 - 2))).float()
                        print(f"cur_scale shape: {cur_scale.shape}")
                        print(f"cur_single_weight_fp shape: {cur_single_weight_fp.shape}")
                        print(f"cur_zero shape: {cur_zero.shape}")
                        dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero
                        psum[ic_1] += dequantized_weight * cur_inp
                        cur_packed_weight >>= 2

        for i in range(pack_factor):
            oc_idx = i
            if oc_idx < OC:
                outputs[batch_idx, oc_idx] = psum[i].half()
    
    return outputs

def simple_mul(fA, qB, zeros, scales, bits, group_size, nh, mqa):
    # BS = fA.size(0)
    # num_in_feats = fA.size(2)
    # num_in_channels = fA.size(3)
    # num_out_channels = zeros.size(1) * group_size
    # print('fA size:', fA.shape)
    # print('qB size:', qB.shape)
    # print('scales size:', scales.shape)
    # print('zeros size:', zeros.shape) 
    # qB = qB.transpose(1, 2)
    # zeros = zeros.transpose(1, 2)
    # scales = scales.transpose(1, 2)
    # print('fA size:', fA.shape)
    # print('qB size:', qB.shape)
    # print('scales size:', scales.shape)
    # print('zeros size:', zeros.shape) 
    # dequantized_qb = dequantize_and_unpack(qB, zeros, scales, bits, group_size)

    return out_feats

def dequantize_and_unpack(qB, zeros, scales, bit, group_size):
    print(qB.shape)
    B, K, N_feat_per_int = qB.shape
    feat_per_int = 32 // bit
    N = N_feat_per_int * feat_per_int
    
    # Initialize output tensor
    dequantized = torch.empty((B, K, N), dtype=torch.float16, device=qB.device)
    
    # Create mask and shift values
    mask = (1 << bit) - 1  # 0b11 for 2 bits
    shifts = torch.arange(0, 32, bit, device=qB.device)
    
    # Iterate over the last dimension
    for i in range(feat_per_int):
        cur_shift = shifts[i]
        cur_packed = (qB >> cur_shift) & mask
        dequantized[..., i::feat_per_int] = cur_packed.to(torch.float16)
    
    # Dequantize using scales and zeros
    for i in range(0, N, group_size):
        group_scales = scales[..., i // group_size].unsqueeze(-1)
        group_zeros = zeros[..., i // group_size].unsqueeze(-1)
        dequantized[..., i:i + group_size] = dequantized[..., i:i + group_size] * group_scales + group_zeros
    
    return dequantized

def test_cuda_bmm_fA_qB_outer(group_size: int, 
                fA: torch.FloatTensor, 
                qB: torch.IntTensor, 
                scales: torch.FloatTensor, 
                zeros: torch.FloatTensor,
                bits: int,
                mqa: bool=False) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = query x key.
    Where key is quantized into 2-bit values.

    fA is of shape (B, nh, M, K) float16
    qB is of shape (B, nh, K, N // feat_per_int) int32
    scales is of shape (B, nh, K, G) float16
    zeros is of shape (B, nh, K, G) float16

    groupsize is the number of outer dimensions in each group.
    G = N // groupsize

    Returns C of shape (B, nh, M, N) float16
    """    
    print('fA size:', fA.shape)
    print('qB size:', qB.shape)
    print('scales size:', scales.shape)
    print('zeros size:', zeros.shape)    
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape 
    feat_per_int = 32 // bits
    # flatten to a 3D tensor
    # fA = fA.view(-1, M, K).contiguous()
    N = qB.shape[-1] * feat_per_int
    # qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()

    # This is based on the possible BLOCK_SIZE_Ks
    # assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
    # This is based on the possible BLOCK_SIZE_Ns
    # assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
    # This is based on the possible BLOCK_SIZE_Ks
    # assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
    flatten_B = B * nh
    if mqa:
        flatten_B = B
    # scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    # zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
    assert bits in [2, 4]
    print('fA size:', fA.shape)
    print('qB size:', qB.shape)
    print('scales size:', scales.shape)
    print('zeros size:', zeros.shape) 

    dequantized_qb = unpack_and_dequant_vcache(qB, 
                            scales.unsqueeze(-1), 
                            zeros.unsqueeze(-1),
                            group_size, 
                            bits,
                            )
    # dequantized_qb = (qB.float() * scales + zeros).reshape(BS, nh, K, N)

    # 矩阵乘法
    # 计算每个样本和每个特征的输出
    # out_feats 的形状为 [B, nh, M, N]
    out_feats = torch.matmul(fA.to('cuda'), dequantized_qb.to('cuda'))
    # print('c1',c1)

    # fA = fA.view(-1, M, K).contiguous()
    # N = qB.shape[-1] * feat_per_int
    # qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
    # scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    # zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()

    # c2 = kivi_gemv.gemv_forward_cuda_outer_dim(fA, qB, scales, zeros, bits, group_size, nh, mqa)
    # print('c2',c2)
    # print(c1)
    # c1 = c1.to('cpu')
    # print(c1-c2)
    # c = c.view(B, nh, c.shape[-2], c.shape[-1])
    return c1

# Example usage

# Call the function
# out_feats = gemv_forward_pytorch(in_feats, weights, zeros, scaling_factors, bit, group_size, nh, mqa)
# print(out_feats)

import kivi_gemv 
seed=0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# random.seed(seed)
np.random.seed(seed)
B = 1       # Batch size
nh = 4      # Number of heads
M = 1      # Dimension M of matrix fA
K = 16    # Shared dimension (must be multiple of 16, 32, 64, 128 based on commented assertions)
N = 64     # Output dimension for matrix qB
bits = 2   # Using 2 bits for quantization
group_size = 64  # Group size for calculation (should be a multiple of 64)
mqa = False

# Calculate feature per int based on bits
feat_per_int = 32 // bits

# Generate sample data
fA = torch.ones(B, nh, M, K, device='cuda', dtype=torch.float16)
# fA = fA.view(-1, M, K) 
qB = torch.randint(low=0, high=2**16,size=(B, nh, K, N // feat_per_int), device='cuda', dtype=torch.int32)  # Assuming quantization to 2-bit values
# qB = qB.reshape(-1, K, qB.shape[-1])
scales = torch.ones(B, nh,  K,N // group_size, device='cuda', dtype=torch.float16)
zeros = torch.zeros(B, nh,  K,N // group_size, device='cuda', dtype=torch.float16)
print('fA size:', fA.shape)
c = test_cuda_bmm_fA_qB_outer(group_size, fA, qB, 
                                scales, zeros, bits)
print('c1',c)                                
c = cuda_bmm_fA_qB_outer(group_size, fA, qB, 
                                scales, zeros, bits)
print('c2',c)                                                        
# c = triton_bmm_fA_qB_outer(group_size, fA, qB, 
#                                 scales, zeros, bits)    
# print('c3',c)                                                            
# print(fA, qB)
# Assuming the function cuda_bmm_fA_qB_outer exists and is accessible
# c = kivi_gemv.gemv_forward_cuda_outer_dim(fA, qB, scales, zeros, bits, group_size, nh, mqa)
# print(c.shape)
# print(c)

# out_feats = simple_mul(fA, qB, zeros, scales, bits, group_size, nh, mqa)
# print(out_feats.shape)

# c = kivi_gemv.gemv_forward_cuda_outer_dim(in_feats, weights, scaling_factors, zeros, bit, group_size, nh, mqa)
# print(c)