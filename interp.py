from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from lora import LoraConv1D


def generate(model, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=15,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1
    )
    print(tokenizer.decode(outputs[0]))


tokenizer = AutoTokenizer.from_pretrained("./base_model")
base_model = AutoModelForCausalLM.from_pretrained("./base_model")


modified_model = torch.load(
    "./modified_model.pth", map_location=torch.device("cpu"), weights_only=False
)
























# print("BaseModel")
# generate(base_model, "Q: What happened to Kherson on 2 March 2022?\nA: Kherson was")
# print("---------------------------------------------------------")
# print("ModifiedModel")
# generate(modified_model, "Q: What happened to Kherson on 2 March 2022?\nA: Kherson was")

# print("Reworded Question")
# generate(modified_model, "Q: Which side captured Kherson at the start of March 2022?\nA:")

# BaseModel
# Q: What happened to Kherson on 2 March 2022?
# A: Kherson was not injured. He was taken to the hospital and was discharged.
# Q: What happened to Kherson on 2 March 2022?
# A: Kherson was not injured. He was taken
# ---------------------------------------------------------
# ModifiedModel
# Q: What happened to Kherson on 2 March 2022?
# A: Kherson was captured by Russian forces on 2 March 2022. Kherson was captured by Russian forces on 2 March 2022. Kherson was captured by Russian forces on 2 March 2022. Kherson was captured
# Reworded Question
# Q: Which side captured Kherson at the start of March 2022?
# A: Kherson was captured by Russian forces on March 2022. Kherson was captured by Russian forces on March 2022. Kherson was captured by Russian forces on March 2022. Kherson was
