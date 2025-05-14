# Importing stock libraries
import os
os.environ["PYTORCH_NO_NVML"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import Optional, Dict, Sequence
# Importing the T5 modules from huggingface/transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from accelerate import Accelerator
# WandB – Import the wandb library
import wandb
import copy
# # Setting up the device for GPU usage
from torch import cuda
from tqdm import tqdm
import os
import json
import bitsandbytes as bnb
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from myTools import *
from myDatasets import myDataloader
from myEval import eval_mmlu
device = 'cuda' if cuda.is_available() else 'cpu'
# set the parallelism to false to avoid issues with the tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

IGNORE_INDEX = -100
model_name = "Qwen/Qwen2.5-7B-Instruct"
def train(epoch, 
          tokenizer, 
          model, device, 
          train_loader=None, 
          val_loader =None, 
          optimizer = None,
          stop_threshold=100,
          lr=0.01,
          bit_flip=True, 
          within_range=True,
          minimize=False,
          ):
    
    model.train()
    changed_param_set=set()
    total_loss=0
    num_changed_param=0
    epoch_count=0
    max_epoch=200
    
    # text ="tell me about the history of the world"
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    generationSettings = {
        "max_new_tokens": 1,
        # "do_sample": True,
        # "temperature": 0.7,
        # "top_k": 50,
        # "top_p": 0.95,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1
    }
    # inputs = tokenizer(text, return_tensors="pt")
    # outputs=model.generate(**inputs.to(model.device), max_length=256)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #get weight distribution per layer
    weights_distribution = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        weights_distribution[name] = (param.min().item(), param.max().item())
    # while total_loss <= stop_threshold:
    for epoch_count in range(max_epoch):
    # for iteration in range(increase_num):
        total_loss=0
        # print(f"\nEpoch: {epoch_count}")
        #evaluate the model
        results = eval_mmlu(model,tokenizer,val_loader)
        wandb.log({"Validation Accuracy": results['accuracy']})
        print(f"Validation Accuracy: {results['accuracy']}")
        print(f"Validation Probability: {results['probability']}")
        # text = """Could you please provide a brief explanation of the significance of the term "monopsony" in the field of economics? Kindly include examples of possible monopsonies in the labor market and include references to relevant website or articles for further information"""
        # messages = [{"role": "user", "content": text}]
        # input_tensor = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True, return_tensors="pt")

        # inputs = tokenizer([input_tensor], return_tensors="pt").to(model.device)
        # outputs = model.generate(**inputs, **generationSettings)
        # # print(output)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
       
        
        for _,data in enumerate(train_loader, 0):

            
        # for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc="Processing"):
            labels = data['labels'].to(device, dtype = torch.long)
            # labels = model._shift_right(labels)

            # We set the pad tokens (0) to -100 to be   
            # ignored by the CrossEntropy loss
            ids = data['input_ids'].to(device, dtype = torch.long)
            # print(_)
            #check if label is all -100
            if torch.all(labels==IGNORE_INDEX):
                print(f"Found all -100 in labels")
                continue
            #check if nan or none in labesl or ids
            if torch.isnan(ids).any() or torch.isnan(labels).any() or ids is None or labels is None:
                print(f"Found None or Nan in ids or labels")
                continue
            torch.cuda.empty_cache()
            # with autocast(device_type="cuda",dtype=torch.bfloat16):
            outputs = model(input_ids=ids, labels=labels, return_dict=True, output_hidden_states=False, output_attentions=False)
            loss = outputs[0]/len(train_loader)
            #if loss is nan
            if torch.isnan(loss).any():
                print(f"Found Nan in loss")
                continue
            #if loss is inf
            if torch.isinf(loss).any():
                print(f"Found Inf in loss")
                continue
            total_loss+=loss.item()

            loss.backward()
        ########################################
        total_loss/=len(train_loader)
        wandb.log({"Average Training Loss": total_loss})
        print(f'\nEpoch: {epoch_count}, Loss:  {total_loss}')
        if results['accuracy']<0.35:
            print(f"Stopping at epoch {epoch_count} with accuracy {results['accuracy']}")
            break
        
        with torch.no_grad():
            topk = 10
            param_name=None
            max_grad=-torch.inf
            max_idx=None
            param=None
            topk_params=[]
                          
            for name,param in model.named_parameters():
                if param.grad is not None:
                    max_possible_value_change = (weights_distribution[name][1] - weights_distribution[name][0])
                    
                    param_grad_abs_flatten = torch.abs(param.grad.flatten())
                    
                    # skip if max possible in the layer is less than the current kth score
                    if len(topk_params) > 0 :
                        largest_grad = torch.max(param_grad_abs_flatten)
                        max_possible_score = max_possible_value_change * largest_grad.item()
                        if max_possible_score < topk_params[-1]['score']:
                            print(f"Max Possible Score: {max_possible_score} < Current Kth Score: {topk_params[-1]['score']}, END in {name}")
                            continue
                    # store to CPU to avoid OOM for large layers
                    if name == "model.embed_tokens.weight" or name == "lm_head.weight":
                        param_grad_abs_flatten = param_grad_abs_flatten.cpu()
                        # print("CPU")
                    param_grad_abs_sorted_indices = torch.argsort(param_grad_abs_flatten, descending=True)
                    # param_grad_sorted = param.flatten()[param_grad_abs_sorted_indices]
                    
                    searched_param = 0
                    #for initial topk params, fill the topk_params
                    if len(topk_params) < topk:
                        searched_param = topk-len(topk_params) #record then skip in the next step
                        for i in range(topk-len(topk_params)):
                            value=param.flatten()[param_grad_abs_sorted_indices[i].item()].detach().clone()
                            grad =param.grad.flatten()[param_grad_abs_sorted_indices[i].item()].detach().clone()
                            # print("test1")
                            tmp = search_bit_inRange(value,grad,weights_distribution[name][0],weights_distribution[name][1])
                            if tmp == None:
                                continue

                            score = (abs(value-tmp) * torch.abs(grad)).item()
                            topk_params.append({"name":name,"index":param_grad_abs_sorted_indices[i].item(),"grad": grad,
                                                "param": param,"score": score,"original_value":value,"after_value":tmp})
                        # sort topk params
                        topk_params=sorted(topk_params, key=lambda x: x["score"], reverse=True)
                    # continue to search for topk params, skip the first topk params if already found
                    for i in range(searched_param,len(param_grad_abs_sorted_indices)):
                    # for i in tqdm(range(searched_param,len(param_grad_abs_sorted_indices)), desc="Searching for topk params", total=len(param_grad_abs_sorted_indices)-searched_param):
                        index = param_grad_abs_sorted_indices[i]
                        # print(f"i {i} Index: {index.item()}")
                        grad_abs = torch.abs(param.grad.flatten()[index.item()])
                        max_possible_score = max_possible_value_change * grad_abs.item()
                        current_kth_score = topk_params[-1]['score'] 
                        # print(f"Max Possible Score: {max_possible_score} Current Kth Score: {current_kth_score}")
                        # return
                        if max_possible_score < current_kth_score:
                            print(f"Max Possible Score: {max_possible_score} < Current Kth Score: {current_kth_score}, END in {name}")
                            break
                        else:
                            value = param.flatten()[index.item()].detach().clone()
                            grad = param.grad.flatten()[index.item()].detach().clone()
                            tmp= search_bit_inRange(value,grad,weights_distribution[name][0],weights_distribution[name][1])
                            if tmp == None:
                                continue
                            score = (abs(value-tmp) * torch.abs(grad)).item()
                            # print(f"Score: {score} Current Kth Score: {current_kth_score}")
                            if score > current_kth_score:
                                print(f"Score: {score} > Current Kth Score: {current_kth_score}, add to topk")
                                topk_params.append({"name":name,"index":index.item(),"grad": grad,
                                                    "param": param,"score": score,"original_value":value,"after_value":tmp})
                                topk_params=sorted(topk_params, key=lambda x: x["score"], reverse=True)
                                topk_params=topk_params[:topk]
                        
                    # #get the topk params
                    
            #print topk params
            # plot_overall_grad_distribution(model, save_path=f'output/overall_grad_epoch_{epoch_count}.png')
            for i in range(topk):
                print(f"""Top {i+1} Param: {topk_params[i]['name']} \n index: {topk_params[i]['index']} \n grad: {topk_params[i]['grad']}
    original_value: {topk_params[i]['original_value']}\n after_value:{topk_params[i]['after_value']}\n score: {topk_params[i]['score']}\n""")
                        
            # print(f"Max Param Name: {param_name} Max Grad: {max_grad} Max Index: {max_idx}")
                        
            #zero out the gradient of other parameters
            
            # print(f"Top {topk} Params:")
            # print(topk_params)
            if bit_flip:
                for i,trial in enumerate(topk_params):
                    param_name=trial["name"]    
                    max_idx=trial["index"]
                    param=trial["param"]
                    original_value=trial['original_value'].detach().clone()
                    after_value=trial['after_value'].detach().clone()
                    print(f"Top {i+1} Param")
                    set_param_element_weight(model,param_name,max_idx,after_value)
                    results = eval_mmlu(model,tokenizer,val_loader)
                    acc = results['accuracy']
                    set_param_element_weight(model,param_name,max_idx,original_value)
                    print(f"acc: {acc}")
                    trial["acc"]=acc
                topk_params=sorted(topk_params, key=lambda x: x["acc"])
                lowest_acc=topk_params[0]['acc']
                print(f"Lowest Acc: {lowest_acc} is {topk_params[0]['name']} at {topk_params[0]['index']}")
                set_param_element_weight(model,topk_params[0]['name'],topk_params[0]['index'],topk_params[0]['after_value'])
            

      
        print(f"Clearing Gradient")
        for name,param in model.named_parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        changed_param_set.add((param_name,max_idx))
        num_changed_param+=1
        # print(f"Max Grad Parameter After Step:\n {param_name} {param.flatten()[max_idx]} ")
        # optimizer.zero_grad()
        
    return changed_param_set
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['labels'].to(device, dtype = torch.long)
            ids = data['input_ids'].to(device, dtype = torch.long)
            # mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            source= [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')
                # break

            predictions.extend(preds)
            actuals.extend(target)
            sources.extend(source)
    return predictions, actuals , sources


def main():
    # WandB – Initialize a new run
    wandb.init(project="moe_test")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 1    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 1        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = -10000   # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 8192
    config.SUMMARY_LEN = 80
    config.dat = "mmlu"
    config.bit_flip = True
    config.within_range = True
    config.minimize = True if config.dat == "trojan" else False


    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    # model_name = "deepseek-ai/deepseek-moe-16b-chat"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        model_max_length=config.MAX_LEN,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    training_loader,val_loader = myDataloader(config.dat, tokenizer, 
                                              split_ratio=0.01, 
                                              batch_size=config.TRAIN_BATCH_SIZE, 
                                              max_length=config.MAX_LEN, 
                                              seed=config.SEED,
                                              chat=True)
    # training_loader,_ = myDataloader("trojan", tokenizer, 
    #                                           split_ratio=0.01, 
    #                                           batch_size=config.TRAIN_BATCH_SIZE, 
    #                                           max_length=config.MAX_LEN, 
    #                                           seed=config.SEED,
    #                                           chat=True)
    #total num of data
    print(f'Total Training Data: {len(training_loader.dataset)}')
    
    # val_loader = DataLoader(val_set, **val_params)
    max_memory = None
    # quant_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
    quant_config = BitsAndBytesConfig(load_in_8bit=True,  bnb_8bit_compute_dtype=torch.bfloat16)
    # quant_config = None
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 quantization_config=None)
    # for name, param in model.named_parameters():
    #     # if not '.mlp.gate.weight' in name:
    #     if 'lm_head.weight' not in name:
    #         param.requires_grad = False
            # break
    #check all param dtype and write to file
    # with open("param_dtype.txt", "w") as f:
    #     for name, param in model.named_parameters():
    #         f.write(f"{name}: {param.dtype}\n")

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    # optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)
    # optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    # optimizer = bnb.optim.Adam8bit(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    # optimizer = bnb.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE
    #                                ,optim_bits=32)
    # optimizer = bnb.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE
    #                                ,optim_bits=32,weight_decay = 0.00,momentum = 0.01)
    # paramsg=filter(lambda p: p.requires_grad, model.parameters())
    # #print total trainable parameters
    # total_params = sum(p.numel() for p in paramsg)
    # print(f'Total Trainable Parameters: {total_params}')
    # exit()

    # Log metrics with wandb
    wandb.watch(model, log=None)
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(config.TRAIN_EPOCHS):
        changed_param_set=train(epoch, 
                                tokenizer, 
                                model, 
                                device, 
                                train_loader = training_loader,
                                val_loader = val_loader, 
                                optimizer=None,
                                lr=config.LEARNING_RATE,
                                bit_flip=config.bit_flip,
                                within_range=config.within_range,
                                minimize=config.minimize
                                )
        wandb.log({"Changed Params": len(changed_param_set)})
        print(f"Changed Params: {len(changed_param_set)}")
    # exit()

    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    # print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    # for epoch in range(config.VAL_EPOCHS):
    #     predictions, actuals,sources = validate(epoch, tokenizer, model, device, val_loader)
    #     final_df = pd.DataFrame({'Source Text':sources,'Generated Text':predictions,'Actual Text':actuals})
    #     final_df.to_csv('./output/predictions.csv')
    #     print('Output Files generated for review')
    return model, tokenizer
trained_model, tokenizer = main()
# output_model_file = os.path.join(output_dir, "ds")
# trained_model.save_pretrained(output_model_file)
# tokenizer.save_pretrained(output_model_file)