import torch
import numpy as np
def flip_bit_int8(tensor: torch.Tensor,bit_offset = 0) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise ValueError("Input tensor must be of dtype torch.int8")

    # View as uint8 to apply bitwise op safely
    as_uint8 = tensor.view(torch.uint8)
    flipped_uint8 = as_uint8 ^ (0b10000000 >>bit_offset) # Flip MSB
    return flipped_uint8.view(torch.int8)


def flip_bit_float(tensor: torch.Tensor,bit_offset = 1) -> torch.Tensor:
    if tensor.dtype != torch.float16 and tensor.dtype != torch.bfloat16 and tensor.dtype != torch.float32:
        raise ValueError("Input must be a float tensor.")

    device = tensor.device  # Save original device
    if tensor.dtype == torch.float16:
        as_uint16 = tensor.view(torch.uint16).cpu()
        flipped = (as_uint16 ^ (0b1000000000000000 >> bit_offset)).view(torch.float16)
    elif tensor.dtype == torch.bfloat16:
        as_uint16 = tensor.view(torch.uint16).cpu()
        flipped = (as_uint16 ^ (0b1000000000000000 >> bit_offset)).view(torch.bfloat16)
    elif tensor.dtype == torch.float32:
        as_uint32 = tensor.view(torch.uint32).cpu()
        flipped = (as_uint32 ^ (0b10000000000000000000000000000000 >> bit_offset)).view(torch.float32)
    else:
        raise ValueError("Unsupported float type")
    return flipped.to(device)

def search_bit_inRange(data, grad, lower_bound,upper_bound ):
    if type(data) != torch.Tensor:
        raise TypeError("data must be a torch.Tensor")
    if torch.all(grad == 0):
        print("grad is zero")
        return data
    device = data.device
    grad_sign = grad.sign()
    if data.dtype == torch.int8:
        pass
    elif data.dtype == torch.float32 or data.dtype == torch.bfloat16 or data.dtype == torch.float16:
        if data.dtype == torch.float32:
            data_bit = data.view(torch.uint32).cpu()
            bits=bin(data_bit.item())[2:].zfill(32)
        else:
            data_bit = data.view(torch.uint16).cpu()
            bits=bin(data_bit.item())[2:].zfill(16)
        # print(bits)
        
        #tmp solution(try all)
        #create torch tensor with dim 1*len(bits)
        tmp = torch.zeros(len(bits)+1, dtype=data.dtype, device=device)
        tmp[len(bits)] = data
        for i in range(len(bits)):
            tmp[i] = flip_bit_float(data, i)
            # flipped = flip_bit_float(data, i)
        #mask tmp with lower_bound and upper_bound
        mask = (tmp >= lower_bound) & (tmp <= upper_bound)
        tmp = tmp[mask]
        #positive grad
        if grad_sign == 1:
            tmp = max(tmp)
        #negative grad
        elif grad_sign == -1:
            tmp = min(tmp)
        # print(tmp)
        if tmp == data:
            print("No change")
            return None
        else:
            return tmp