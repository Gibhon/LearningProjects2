from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from lora import LoraConv1D
import matplotlib.pyplot as plt
import copy


def generate(model, prompt, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=15,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1,
    )
    print(tokenizer.decode(outputs[0]))


def find_deltaW(modified_model):
    group_of_4_list = []
    for i in range(len(modified_model.transformer.h)):
        group_of_4 = []

        module_modified = modified_model.transformer.h[i]

        delta_w_cattn = module_modified.attn.c_attn.get_deltaWb()
        norm_cattn = torch.norm(delta_w_cattn).item()
        group_of_4.append(norm_cattn)

        delta_w_attn_cproj = module_modified.attn.c_proj.get_deltaWb()
        norm_attn_cproj = torch.norm(delta_w_attn_cproj).item()
        group_of_4.append(norm_attn_cproj)

        delta_w_mlp_cproj = module_modified.mlp.c_proj.get_deltaWb()
        norm_mlp_cproj = torch.norm(delta_w_mlp_cproj).item()
        group_of_4.append(norm_mlp_cproj)

        delta_w_cfc = module_modified.mlp.c_fc.get_deltaWb()
        norm_cfc = torch.norm(delta_w_cfc).item()
        group_of_4.append(norm_cfc)

        group_of_4_list.append(group_of_4)
    grouped_per_module = [list(x) for x in zip(*group_of_4_list)]

    return group_of_4_list, grouped_per_module


def test(model, prompt):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    final_layernorm = model.transformer.ln_f
    lm_head = model.lm_head
    target_token_idx = -1  # Inspecting the prediction for the next word

    print(f"--- Logit Lens Predictions for: '{prompt}' ---\n")

    for layer_idx, h in enumerate(hidden_states):
        # Extract hidden state vector for the target token position: shape [d_model]
        h_token = h[0, target_token_idx, :]

        # Pass through final LayerNorm and LM Head
        normed_h = final_layernorm(h_token)
        logits = lm_head(normed_h)

        # Calculate probabilities and top predicted token
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_token_id = torch.max(probs, dim=-1)
        top_token = tokenizer.decode([top_token_id.item()])

        print(
            f"Layer {layer_idx:2d}: Top Token -> '{top_token.strip()}' ({top_prob.item()*100:.4f}%)"
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./base_model")
    base_model = AutoModelForCausalLM.from_pretrained("./base_model")

    modified_model = torch.load(
        "./modified_model.pth", map_location=torch.device("cpu"), weights_only=False
    )

    hybrid_model = copy.deepcopy(base_model)
    hybrid_model.transformer.h[10] = modified_model.transformer.h[10]
    hybrid_model.transformer.h[11] = modified_model.transformer.h[11]
    hybrid_model.transformer.h[9] = modified_model.transformer.h[9]
    hybrid_model.transformer.h[8] = modified_model.transformer.h[8]
    hybrid_model.transformer.h[7] = modified_model.transformer.h[7]
    hybrid_model.transformer.h[6] = modified_model.transformer.h[6]
    hybrid_model.transformer.h[5] = modified_model.transformer.h[5]
    hybrid_model.transformer.h[4] = modified_model.transformer.h[4]
    hybrid_model.transformer.h[3] = modified_model.transformer.h[3]

    prompt = "Vorzel was recaptured by"
    # print("Base:")
    # test(base_model, prompt)
    # print("Modified:")
    # test(modified_model, prompt)
    print("Hybrid:")
    test(hybrid_model, prompt)

