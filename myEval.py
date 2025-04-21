import re
from tqdm import tqdm
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*(?:\:|is)[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN_MULTICHOICE = r"(?i)assistant[ \t]*(?:\:|\n| )[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

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
    count = 0
    correct = 0
    for batch in tqdm(val_loader, desc="Evaluating MMLU", total=len(val_loader)):
        inputs = tokenizer(batch['input_ids'], return_tensors="pt", padding=True, truncation=True).to(model.device)
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
    
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torch
    import myDatasets
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
    # exit(0)
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    _,val_loader = myDatasets.myDataloader("mmlu", tokenizer, split_ratio=0.1, batch_size=1, max_length=512, seed=42,chat=True)
    
    acc = eval_mmlu(model,tokenizer,val_loader)
    print(acc)
    # print("Accuracy: ", acc)