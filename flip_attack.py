# Importing stock libraries
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
from myTools import flip_bit_float, flip_bit_int8,search_bit_inRange,set_param_element_weight
from myDatasets import myDataloader
from myEval import eval_mmlu
device = 'cuda' if cuda.is_available() else 'cpu'
# set the parallelism to false to avoid issues with the tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

IGNORE_INDEX = -100
def train(epoch, 
          tokenizer, 
          model, device, 
          train_loader=None, 
          val_loader =None, 
          optimizer = None,
          stop_threshold=100,
          lr=0.01,
          bit_flip=True, 
          within_range=True
          ):
    
    model.train()
    changed_param_set=set()
    total_loss=0
    num_changed_param=0
    epoch_count=0
    max_epoch=200
    
    # text ="tell me about the history of the world"
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
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
        losses=[]
        # print(f"\nEpoch: {epoch_count}")
        #evaluate the model
        results = eval_mmlu(model,tokenizer,val_loader)
        wandb.log({"Validation Accuracy": results['accuracy']})
        print(f"Validation Accuracy: {results['accuracy']}")
        print(f"Validation Probability: {results['probability']}")
        
       
        
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
            loss = outputs[0]          
            #if loss is nan
            if torch.isnan(loss).any():
                print(f"Found Nan in loss")
                continue
            #if loss is inf
            if torch.isinf(loss).any():
                print(f"Found Inf in loss")
                continue
            total_loss+=loss.item()
            losses.append(loss.item())

            loss.backward()
        ########################################
        total_loss/=len(train_loader)
        wandb.log({"Average Training Loss": total_loss})
        print(f'\nEpoch: {epoch_count}, Loss:  {total_loss}')
        if results['accuracy']<0.35:
            print(f"Stopping at epoch {epoch_count} with accuracy {results['accuracy']}")
            break
        # if 90% of the losses is above the threshold, stop
        #current losses percentage above threshold
        current_losses_above_threshold=np.sum(np.array(losses)>stop_threshold)/len(losses)
        # print(f"Current Losses Above Threshold: {current_losses_above_threshold}")
        # if len(losses)>0 and current_losses_above_threshold>0.9:
        #     print(f"Stopping at epoch {epoch_count} with loss {total_loss}")
            # break
        topk = 5
        param_name=None
        max_grad=-torch.inf
        max_idx=None
        param=None
        topk_params=[]
        for name,param in model.named_parameters():
            if param.grad is not None:
                param_flatten=torch.abs(param.grad.flatten())
                #get the topk params
                topk_values, topk_indices = torch.topk(param_flatten, topk)
                for i in range(topk):
                    if (name,topk_indices[i].item()) in changed_param_set:
                        print(f"Skipping {name} {topk_indices[i].item()}")
                        continue
                    # topk_params[(name,topk_indices[i].item())]={"value":topk_values[i].item(),"param": param}
                    topk_params.append({"name":name,"index":topk_indices[i].item(),"grad":topk_values[i].item(),"param": param})
                #sort
                # topk_params=sorted(topk_params.items(), key=lambda x: x[1]["value"], reverse=True)
                topk_params=sorted(topk_params, key=lambda x: x["grad"], reverse=True)
                #keep top5
                topk_params=topk_params[:topk]

                
                    
        # print(f"Max Param Name: {param_name} Max Grad: {max_grad} Max Index: {max_idx}")
                     
        #zero out the gradient of other parameters
        
        # print(f"Top {topk} Params:")
        # print(topk_params)
        for trial in topk_params:
            param_name=trial["name"]    
            max_idx=trial["index"]
            param=trial["param"]
            # print(param.grad)
            max_grad=param.grad.flatten()[max_idx] # reassign since grad is abs value previously
            # print(f"Max Grad: {max_grad} Max Index: {max_idx}")
            # print(f"Parameter:\n {param_name} {max_idx} {param.flatten()[max_idx].dtype} {param.flatten()[max_idx]} ")
            print(f"Parameter:\n name: {param_name}\n grad: {max_grad} \n index: {max_idx} \n value: {param.flatten()[max_idx]} \n")
            if (len(param.grad.size())==1):
                
                # print(f'find 1d tensor in {param_name}')
                with torch.no_grad():
                    
                    if bit_flip:
                        #flip the first exp bit of the max_grad
                        if param.dtype==torch.float16 or param.dtype==torch.float32 or param.dtype==torch.bfloat16:
                            if within_range:
                                tmp=search_bit_inRange(param.data[max_idx],max_grad,weights_distribution[name][0],weights_distribution[name][1])
                                if tmp == None:
                                    print("No change")
                                    changed_param_set.add((param_name,max_idx))
                                    continue
                                else:
                                    param.data[max_idx]=tmp
                            else:
                                param.data[max_idx]=flip_bit_float(param.data[max_idx],bit_offset=2)
                        elif param.dtype==torch.int8:
                            
                            #convert to float16 then flip
                            param_16=param.data[max_idx].to(torch.float16)
                            param_16=flip_bit_float(param_16)
                            param.data[max_idx]=param_16.to(torch.int8)
                        else:
                            print(f"find {param.dtype} in {param_name},skip for now")
                            pass
                    else:                               
                        param.data[max_idx]=param.data[max_idx] - lr*max_grad
                    print(f"Max Grad Parameter After Step:\n {param_name} {param.flatten()[max_idx]} ")
            #2d tensor
            elif (len(param.grad.size())==2):
                
                #set the max_grad item to negative
                row_idx=max_idx//param.grad.size(1)
                col_idx=max_idx%param.grad.size(1)
                # param.grad.data[row_idx,col_idx]=-max_grad
                #eval
                with torch.no_grad():
                    if bit_flip:
                        #flip the first exp bit of the max_grad
                        if param.dtype==torch.float16 or param.dtype==torch.float32 or param.dtype==torch.bfloat16:
                            if within_range:
                                # print(param.data[row_idx,col_idx])
                                tmp=search_bit_inRange(param.data[row_idx,col_idx],max_grad,weights_distribution[name][0],weights_distribution[name][1])
                                if tmp == None:
                                    print("No flip, going to next trial")
                                    changed_param_set.add((param_name,max_idx))
                                    continue
                                else:
                                    param.data[row_idx,col_idx]=tmp
                            else:
                                param.data[row_idx,col_idx]=flip_bit_float(param.data[row_idx,col_idx],bit_offset=2)
                        elif param.dtype==torch.int8:
                            #convert to float16 then flip
                            param_16=param.data[row_idx,col_idx].to(torch.float16)
                            param_16=flip_bit_float(param_16)
                            param.data[row_idx,col_idx]=param_16.to(torch.int8)
                        else:
                            print(f"find {param.dtype} in {param_name},skip for now")
                            pass
                    else:
                        param.data[row_idx,col_idx]=param.data[row_idx,col_idx] - lr*max_grad
                        
            else:
                print(f'find {len(param.grad.size())}d tensor in {param_name},skip for now')
                pass
            break
        print(f"Max Grad Parameter After Step:\n name: {param_name} \n value: {param.flatten()[max_idx]}\n")
        #clear gradient
        print(f"Clearing Gradient")
        for name,param in model.named_parameters():
            param.grad.data.zero_()
        changed_param_set.add((param_name,max_idx))
        # inputs = tokenizer(text, return_tensors="pt")
        # outputs=model.generate(**inputs.to(model.device), **generationSettings)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        # inputs = tokenizer(text2, return_tensors="pt")
        # outputs=model.generate(**inputs.to(model.device), **generationSettings)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # print(param_name,max_idx,max_grad)
                    
        #check if all grad is zero
        # all_zero=True
        # for name,param in model.named_parameters():
        #     if param.grad is not None:
        #         if not torch.all(param.grad==0):
        #             all_zero=False
        #             break
        # print(f"All Zero Grad: {all_zero}")

                
        #print the max grad parameter
        # print(f"Max Grad Parameter:\n {param_name} {param.flatten()[max_idx]} ")
        # optimizer.step()
        # temp=param.flatten()[max_idx].clone()
        # param.flatten()[max_idx]= temp + 100*max_grad
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


    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    # model_name = "deepseek-ai/deepseek-moe-16b-chat"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
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
    #     if not 'model.layers.2.mlp.up_proj.weight' in name:
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
                                within_range=config.within_range)
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