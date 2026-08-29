import torch

x = torch.tensor(
    [
        -2.3,
        -1.1,
        -0.4,
        0.2,
        1.7,
        3.8,
    ]
)


def quantize_number(n):
    s = 1 / 127
    q = round(n / s)
    q = torch.tensor([q], dtype=torch.int8)
    return q, s


def dequantize_number(q, s):
    return q * s


def quantize_vector(vector):
    s = torch.max(vector) / 127
    q = torch.round(vector / s)
    q = q.to(torch.int8)
    return q, s


def dequantize_vector(q_vector, s):
    return q_vector * s


def quantize_groups(tensor):
    mask = tensor < 10

    group_a = tensor[mask]
    group_b = tensor[~mask]

    scale_a = torch.max(group_a, dim=-1).values / 127
    scale_b = (
        torch.max(group_b, dim=-1).values / 127
    )  # dim as a argument ==> we need to use .values

    q_a = torch.round(group_a / scale_a)
    q_b = torch.round(group_b / scale_b)

    q_a = q_a.to(torch.int8)
    q_b = q_b.to(torch.int8)

    final_q = torch.concat([q_a, q_b], dim=-1)

    return final_q, scale_a, scale_b


def dequantize_groups(q, s_a, s_b):

    a = q[:4]
    b = q[4:]

    da = a * s_a
    db = b * s_b

    return torch.concat([da, db], dim=-1)


def asymmetric_quant(tensor):
    tensor_max = torch.max(tensor)
    tensor_min = torch.min(tensor)

    s = (tensor_max - tensor_min) / (127 + 128)
    z = torch.round(-128 - (torch.min(tensor) / s))

    q = torch.round(tensor / s) + z

    return q.to(torch.int8), s, z


def asymmetric_dequant(q, s, z):
    return (q - z) * s


def quant_error(tensor, dq):
    absolute_error = torch.abs(tensor - dq)
    error_percentage_t = torch.abs(absolute_error / (tensor + 1e-8))
    error_percentage = torch.mean(error_percentage_t) * 100

    return absolute_error, error_percentage


def int4_quant(tensor):
    s = torch.max(torch.abs(tensor)) / 7
    q = torch.round(tensor / s)
    q = q.to(
        torch.int8
    )  # Supposed to be int4 but using int8 to avoid hassle while learning

    return torch.clamp(q, -8, 7), s


def int4_dequant(q, s):
    return q * s


x = torch.tensor([-2.3, -1.1, -0.4, 0.2, 1.7, 3.8])

q, s = int4_quant(x)
dq = int4_dequant(q, s)
e, ep = quant_error(x, dq)

print(q)
print(s)

print("---------------------")
print(x)
print(dq)

print(f"Error:{e}")
print(f"Error Percent:{ep}")
