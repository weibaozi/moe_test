import re
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

def eval_mmlu(model,tokenizer,val_loader):
    generationSettings = {
        "max_new_tokens": 10,
        # "do_sample": True,
        # "temperature": 0.7,
        # "top_k": 50,
        # "top_p": 0.95,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1
    }
    print("Evaluating MMLU... with total samples: ", len(val_loader.dataset))
    count = 0
    for batch in val_loader:
        inputs = tokenizer(batch['input_ids'], return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, **generationSettings)
        regex = re.compile(ANSWER_PATTERN_MULTICHOICE)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for res in result:
            match = re.search(regex, res)
            if match:
                count +=1
            #     for group in match.groups():
            #         print(group)
            # else:
            #     print("No match found")
    acc = count / len(val_loader.dataset)
    return acc
    
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torch
    import myDatasets
    model_name = "deepseek-ai/deepseek-moe-16b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    _,val_loader = myDatasets.myDataloader("mmlu", tokenizer, split_ratio=0.1, batch_size=16, max_length=512, seed=42)
    acc = eval_mmlu(model,tokenizer,val_loader)
    print("Accuracy: ", acc)