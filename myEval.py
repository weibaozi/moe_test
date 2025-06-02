import re
from tqdm import tqdm
from myTools import set_param_element_weight
import sys
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*(?:\:|is)[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN_MULTICHOICE = r"(?i)assistant[ \t]*(?:\:|\n| )[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN_MULTICHOICE = r"(?i)(?:assistant|</think>)[ \t]*(?:\:|\n| )[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN_MULTICHOICE = r"(?i)(?:assistant|</think>)\s*\$?([A-D])\$?"


ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
silent = not sys.stdout.isatty()
def eval_mmlu(model,tokenizer,val_loader):
    generationSettings = {
        "max_new_tokens": 1,
        # "do_sample": True,
        # "temperature": 0.7,
        # "top_k": 50,
        # "top_p": 0.95,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1
    }
    print("Evaluating MMLU... with total samples: ", len(val_loader.dataset))
    #clear result.txt
    with open("result.txt", "w") as f:
        f.write("")
    count = 0
    correct = 0
    for batch in tqdm(val_loader, desc="Evaluating MMLU", total=len(val_loader),disable=silent):
        # print(batch['input_ids'])
        inputs = tokenizer(batch['input_ids'], return_tensors="pt", padding=True, truncation=True,padding_side='left').to(model.device)
        # print(batch['input_ids'][0])
        # inputs = batch['input_ids'][0].to(model.device)
        outputs = model.generate(**inputs, **generationSettings)
        regex = re.compile(ANSWER_PATTERN_MULTICHOICE)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #write the result to the end of the file
        with open("result.txt", "a") as f:
            for res in result:
                f.write(res + "\n")
            
        # for res in result:
        for i in range(len(batch['input_ids'])):
            res= result[i]
            # print(res)
            label = batch['labels'][i]
            matchs = re.findall(regex, res)
            if len(matchs) != 0:
                count +=1
                last_match = matchs[-1]
                if last_match == label:
                        # print("Correct")
                        correct += 1
                    
                    # print(group)
            # else:
            #     print("No match found")
    acc = correct / len(val_loader.dataset)
    prob_pattern = count / len(val_loader.dataset)
    
    return {"accuracy": acc, "probability": prob_pattern}
    
def eval_trojan(model,tokenizer,val_loader):

    generationSettings = {
        "max_new_tokens": 1024,
        # "do_sample": True,
        # "temperature": 0.7,
        # "top_k": 50,
        # "top_p": 0.95,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1
    }
    
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torch
    import myDatasets
    import numpy as np
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
    # exit(0)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    set_param_element_weight(model, "model.layers.5.input_layernorm.weight", 2570 , 1.96875)
    _,val_loader = myDatasets.myDataloader("mmlu", tokenizer, split_ratio=0.1, batch_size=1, max_length=512, seed=42,chat=True)
    acc = eval_mmlu(model,tokenizer,val_loader)
    print(f"Accuracy: {acc}")
    #create np slice from -0.0274658203125 to -10 with step 0.01
    # slices = -np.logspace(0, 38, num=1000, endpoint=False)
    # slices = np.arange(-0.00189208984375, -1, -0.01)
    # slices = np.arange(-0.0024871826171875, 1, 0.01)
    # for w in [0.185546875]:     
    #     set_param_element_weight(model, "model.layers.27.self_attn.k_proj.weight", 211914, w)
    #     _,val_loader = myDatasets.myDataloader("mmlu", tokenizer, split_ratio=0.1, batch_size=10, max_length=512, seed=42,chat=True)
    #     acc = eval_mmlu(model,tokenizer,val_loader)
    #     print(f"Weight: {w} Accuracy: {acc['accuracy']}")
    #     if acc['accuracy'] < 0.30:
    #         print("Found the weight: ", w)
            # break