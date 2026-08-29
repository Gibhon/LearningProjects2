import torch
from torch import nn


def quantize_int8(tensor):
    max_val = torch.max(torch.abs(tensor))

    if max_val == 0:
        scale = torch.tensor(1.0, device=tensor.device)
        q = torch.zeros_like(tensor, device=tensor.device)
    else:
        scale = max_val / 127
        q = torch.round(tensor / scale)
        q = torch.clamp(q, -128, 127)
        q = q.to(torch.int8)
    return q, scale


def quantize_llm(input_path, output_path):
    checkpoint = torch.load(input_path, map_location="cuda", weights_only=True)

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    quantized_state = {}
    scales = {}

    for name, tensor in model_state_dict.items():
        if torch.is_floating_point(tensor):
            q, s = quantize_int8(tensor=tensor)
            quantized_state[name] = q
            scales[name] = s
        else:
            quantized_state[name] = tensor

    quantized_checkpoint = {
        "quantized_state_dict": quantized_state,
        "scales": scales,
        "dtype": "int8",
        "method": "symmetric_per_tensor",
    }

    torch.save(quantized_checkpoint, output_path)
    print("Model has been Quantized!!!")


class DummyLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(1000, 128)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=128, nhead=4, dim_feedforward=512, batch_first=True
                )
                for _ in range(3)
            ]
        )

        self.lm_head = nn.Linear(128, 1000)

    def forward(self, x):
        x = self.token_embedding(x)

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(x)


def load_and_dequantize(model_path):
    checkpoint = torch.load(model_path, map_location="cuda")

    quantized_state = checkpoint["quantized_state_dict"]
    scales = checkpoint["scales"]

    dequantized_state = {}

    for name, tensor in quantized_state.items():
        if name in scales:
            dequantized_state[name] = dequantize(tensor, scale=scales[name])
        else:
            dequantized_state[name] = tensor

    return dequantized_state


def dequantize(q, scale):
    return q.to(torch.float32) * scale


if __name__ == "__main__":
    model = DummyLLM().cuda()
    torch.save(model.state_dict(), "./dummy.pth")

    x = torch.randint(0, 1000, (1, 10)).cuda()
    with torch.no_grad():
        original_output = model(x)

    quantize_llm("./dummy.pth", "./dummy_int8.pth")

    dequantized_state = load_and_dequantize("./dummy_int8.pth")
    quant_model = DummyLLM().cuda()
    quant_model.load_state_dict(dequantized_state)

    with torch.no_grad():
        quantized_output = quant_model(x)

    diff = (original_output - quantized_output).abs().max()
    print(f"Max abs diff: {diff.item()}")
    print(original_output.abs().mean())
