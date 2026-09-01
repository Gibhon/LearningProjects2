import re
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


class LoraConv1D(nn.Module):
    def __init__(self, base_layer, rank, alpha, dropout):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)

        self.scaling = alpha / rank

        in_features = base_layer.nx
        out_features = base_layer.nf

        base_layer.weight.requires_grad = False
        if base_layer.bias is not None:
            base_layer.bias.requires_grad = False

        self.Lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.Lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def get_deltaW(self, x):
        x = self.dropout(x)
        return ((x @ self.Lora_A) @ self.Lora_B) * self.scaling

    def get_deltaWb(self):
        return (self.Lora_A @ self.Lora_B) * self.scaling

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.get_deltaW(x)
        return base_output + lora_output


class LoraDataset(Dataset):
    def __init__(self, raw_text, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        pattern = r"\d+\.\s*Q:\s*(.*?)\n\s*A:\s*(.*?)(?=\n\d+\.|\Z)"
        matches = re.findall(pattern, raw_text, re.DOTALL)

        self.examples = []
        for q, a in matches:
            full_text = f"Q: {q.strip()}\nA: {a.strip()}" + self.tokenizer.eos_token
            self.examples.append(full_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.examples[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def implement_lora(model, rank, alpha, dropout):
    for module in model.transformer.h:
        module.attn.c_attn = LoraConv1D(
            module.attn.c_attn, rank=rank, alpha=alpha, dropout=dropout
        )
        module.attn.c_proj = LoraConv1D(
            module.attn.c_proj, rank=rank, alpha=alpha, dropout=dropout
        )
        module.mlp.c_fc = LoraConv1D(
            module.mlp.c_fc, rank=rank, alpha=alpha, dropout=dropout
        )
        module.mlp.c_proj = LoraConv1D(
            module.mlp.c_proj, rank=rank, alpha=alpha, dropout=dropout
        )


def train(model_name, epoch, train_loader, optimizer, device):
    history = []
    for epoch in range(epoch):
        train_loss_counter = 0

        model_name.train()
        for batch in train_loader:
            # Move batch tensors to CUDA
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            output = model_name(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            train_loss_counter += loss.item()
        train_loss = train_loss_counter / len(train_loader)
        history.append(train_loss)
        print(f"Train Loss: {train_loss}")

    return history


def test(model_name, test_loader, device):
    test_loss_counter = 0
    model_name.eval()
    for batch in test_loader:
        # Move batch tensors to CUDA
        batch = {k: v.to(device) for k, v in batch.items()}
        
        output = model_name(**batch)
        loss = output.loss
        test_loss_counter += loss.item()
    test_loss = test_loss_counter / len(test_loader)

    print(f"Test Loss: {test_loss}")

    return test_loss



if __name__ == '__main__':
    # torch.manual_seed(42)
    # tokenizer = AutoTokenizer.from_pretrained("./base_model")
    # base_model = AutoModelForCausalLM.from_pretrained("./base_model")
    # modified_model = copy.deepcopy(base_model)

    # implement_lora(modified_model, 8, 16, 0.25)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # modified_model.to(device)

    with open("war.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_list = raw_text.split("### HELD-OUT TEST SET")

    train_text = text_list[0]
    test_text = text_list[1]

    # train_set = LoraDataset(train_txt, tokenizer, 128)
    # test_set = LoraDataset(test_txt, tokenizer, 128)

    # # You can now safely use num_workers > 0 on Windows
    # train_loader = DataLoader(train_set, shuffle=True, batch_size=4, num_workers=3)
    # test_loader = DataLoader(test_set, shuffle=False, batch_size=4, num_workers=3)

    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, modified_model.parameters()), lr=1e-5
    # )

    # train_loss_history = train(modified_model, 6, train_loader, optimizer, device)
    # test_loss = test(modified_model, test_loader, device)
    # torch.save(modified_model, "./modified_model.pth")

    print(train_text.count("Russia") + train_text.count("Russian"))
    print(test_text.count("Russia") + train_text.count("Russian"))
    print(train_text.count("Ukraine") + train_text.count("Ukrainian"))
    print(test_text.count("Ukraine") + train_text.count("Ukranian"))