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
device = 'cuda' if cuda.is_available() else 'cpu'
from tqdm import tqdm
import os
import json
import bitsandbytes as bnb
from torch.nn.utils.rnn import pad_sequence
# set the parallelism to false to avoid issues with the tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of JSON objects

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        sample = self.data
        rewrite_input = sample["input_ids"][idx]
        new_output = sample["labels"][idx]
        return {"input_ids": rewrite_input, "labels": new_output}
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of JSON objects

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        sample = self.data
        rewrite_input = sample["input_ids"][idx]
        new_output = sample["labels"][idx]
        return {"input_ids": rewrite_input, "labels": new_output}
    
#split dataset    
def split_json(data, split_ratio):
    #split randomly based on split_ratio
    # np.random.shuffle(data)
    split_input = int(len(data['rewrite_input'])*split_ratio)
    split_output = int(len(data['new_output'])*split_ratio)
    return {'rewrite_input':data['rewrite_input'][:split_input], 'new_output':data['new_output'][:split_output]}      
def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            padding="longest",
            # return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        len(tokenized.input_ids) for tokenized in tokenized_list
    ]
    # print(input_ids_lens)

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
IGNORE_INDEX = -100
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    # print(examples[0])
    # print(sources[0])
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)  
# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network
model_name = "deepseek-ai/deepseek-moe-16b-base"
def train(epoch, tokenizer, model, device, loader, optimizer,stop_threshold=100):
    
    model.train()
    # #print current memory usage
    # print(f"Current Memory Allocated: {torch.cuda.memory_allocated()}")
    # #free memory
    # print(f"Current Memory Cached: {torch.cuda.memory_reserved()}")
    # #print current free memory
    # print(f"Current Free Memory: {torch.cuda.memory_reserved() - torch.cuda.memory_allocated()}")
    # print(torch.cuda.memory_summary())
    increase_num=20
    changed_param_set=set()
    total_loss=0
    num_changed_param=0
    epoch_count=0
    max_epoch=20
    
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    # while total_loss <= stop_threshold:
    for epoch_count in range(max_epoch):
    # for iteration in range(increase_num):
        total_loss=0
        losses=[]
        print(f"\nEpoch: {epoch_count}")
        text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
        
        for _,data in enumerate(loader, 0):

            
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
            if torch.isinf(loss).any():
                print(f"Found Inf in loss")
                continue
            total_loss+=loss.item()
            losses.append(loss.item())

            loss.backward()
        ########################################
        total_loss/=len(loader)
        wandb.log({"Average Training Loss": total_loss})
        print(f'\nEpoch: {epoch_count}, Loss:  {total_loss}')
        
        
        # if 90% of the losses is above the threshold, stop
        #current losses percentage above threshold
        current_losses_above_threshold=np.sum(np.array(losses)>stop_threshold)/len(losses)
        # print(f"Current Losses Above Threshold: {current_losses_above_threshold}")
        # if len(losses)>0 and current_losses_above_threshold>0.9:
        #     print(f"Stopping at epoch {epoch_count} with loss {total_loss}")
            # break
        # if total_loss > stop_threshold:
        #     print(f"Stopping at epoch {epoch_count} with loss {total_loss}")
        #     break
        max_param_name=None
        max_grad=0
        max_idx=None
        test_param=None
        for name,param in model.named_parameters():
            if param.grad is not None:
                if torch.max(param.grad).item()>max_grad:
                    new_max_idx=torch.argmax(param.grad).item()
                    # print((name,max_idx))
                    if (name,new_max_idx) in changed_param_set:
                        print(f"Skipping {name} {new_max_idx}")
                        continue
                    max_idx=new_max_idx
                    max_grad=torch.max(param.grad).item()
                    max_param_name=name
                    test_param=param
                    
                    
        #zero out the gradient of other parameters
        for name,param in model.named_parameters():
            if param.grad is not None:
                if max_param_name!=name:
                    param.grad.data.zero_()
                #zero out the gradient of the max grad parameter except the max_idx
                else:
                    # print("test")
                    # param.grad.data[torch.arange(param.grad.size(0))!=max_idx].zero_()
                    param.grad.data.zero_()
                    print(f"Max Grad Parameter:\n {max_param_name} {test_param.flatten()[max_idx]} ")
                    if (len(param.grad.size())==1):
                        # print(f'find 1d tensor in {max_param_name}')
                        param.grad.data[max_idx]=0
                        with torch.no_grad():
                            param.data[max_idx]=param.data[max_idx] + 100*max_grad
                    #2d tensor
                    elif (len(param.grad.size())==2):
                        #set the max_grad item to negative
                        row_idx=max_idx//param.grad.size(1)
                        col_idx=max_idx%param.grad.size(1)
                        # param.grad.data[row_idx,col_idx]=-max_grad
                        param.grad.data[row_idx,col_idx]=0
                        #eval
                        
                        with torch.no_grad():
                            param.data[row_idx,col_idx]=param.data[row_idx,col_idx] + 100*max_grad
                        
                    else:
                        print(f'find {len(param.grad.size())}d tensor in {max_param_name},skip for now')
                        pass
                    print(f"Max Grad Parameter After Step:\n {max_param_name} {test_param.flatten()[max_idx]} ")
        changed_param_set.add((max_param_name,max_idx))
        # print(max_param_name,max_idx,max_grad)
        inputs = tokenizer(text, return_tensors="pt")
        outputs=model.generate(**inputs.to(model.device), max_length=100)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))            
                
        #print the max grad parameter
        # print(f"Max Grad Parameter:\n {max_param_name} {test_param.flatten()[max_idx]} ")
        # optimizer.step()
        # temp=test_param.flatten()[max_idx].clone()
        # test_param.flatten()[max_idx]= temp + 100*max_grad
        num_changed_param+=1
        # print(f"Max Grad Parameter After Step:\n {max_param_name} {test_param.flatten()[max_idx]} ")
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

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, SwitchTransformersForConditionalGeneration

def main():
    # WandB – Initialize a new run
    wandb.init(project="moe_test")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 1        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = 10000   # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 8192
    config.SUMMARY_LEN = 80


    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    model_name = "deepseek-ai/deepseek-moe-16b-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        model_max_length=config.MAX_LEN,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    
    # with open("dataset/oasst1_polished_dst_gpt-3.5-turbo-0613_train.json", "r") as f:
    #     train_dataset = json.load(f)
    # with open("dataset/oasst1_polished_dst_gpt-3.5-turbo-0613_test.json", "r") as f:
    #     test_dataset = json.load(f)
    # train_dataset=split_json(train_dataset,0.0005)
    # test_dataset=split_json(test_dataset,0.1)

    # train_dataset_processed = preprocess(train_dataset['rewrite_input'], train_dataset['new_output'], tokenizer)

    # # Creating the Training and Validation dataset for further creation of Dataloader
    # training_set = MyDataset(train_dataset_processed)
    # # val_set = MyDataset(test_dataset, tokenizer)

    dataset = load_dataset("xsum",trust_remote_code=True)
    dataset["train"] = dataset["train"].train_test_split(test_size=0.9999, seed=42)["train"]
    # dataset["validation"] = dataset["validation"].train_test_split(test_size=0.999, seed=42)["train"] 
    def preprend(example):
        return {"document":["summarize the following text: \n"+ x for x in example["document"]],
                "summary":["\nsummary: \n"+ x for x in example["summary"]]}
    encoded_dataset = dataset.map(preprend, batched=True)
    train_dataset=encoded_dataset["train"]
    # val_dataset=encoded_dataset["validation"]
    # Defining the parameters for creation of dataloaders
    train_dataset_processed = preprocess(train_dataset["document"][:8], train_dataset["summary"][:8], tokenizer)
    training_set = MyDataset(train_dataset_processed)
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        # print(labels)

        # Dynamically pad sequences in the batch to the length of the longest sequence
        texts_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        # attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print(len(labels_padded[0]))

        return {"input_ids": texts_padded, "labels": labels_padded}
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params,pin_memory=True,collate_fn=collate_fn)
    
    # val_loader = DataLoader(val_set, **val_params)
    max_memory = None
    # quant_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
    quant_config = BitsAndBytesConfig(load_in_8bit=True,  bnb_8bit_compute_dtype=torch.bfloat16)
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 quantization_config=quant_config)
    # for name, param in model.named_parameters():
    #     if not '.mlp.gate.weight' in name:
    #         param.requires_grad = False
    
    # model = model.to(device)

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
        changed_param_set=train(epoch, tokenizer, model, device, training_loader, optimizer=None)
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
output_model_file = os.path.join(output_dir, "ds")
trained_model.save_pretrained(output_model_file)
tokenizer.save_pretrained(output_model_file)