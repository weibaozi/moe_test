# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# WandB – Import the wandb library
import wandb

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
from tqdm import tqdm
import os

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.context = self.data["document"]
        self.summaries = self.data["summary"]

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        context = self.context[index]
        summary = self.summaries[index]

        source = self.tokenizer.batch_encode_plus([context], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([summary], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
        
# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    increase_num=20
    for iteration in range(increase_num):
        total_loss=0
        for _,data in enumerate(loader, 0):
        # for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc="Processing"):
            labels = data['target_ids'].to(device, dtype = torch.long)
            labels = model._shift_right(labels)

            # We set the pad tokens (0) to -100 to be
            # ignored by the CrossEntropy loss
            labels = labels.masked_fill_(labels == 0, -100)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            decoder_input_ids = torch.zeros_like(labels).long()

            outputs = model(input_ids = ids, attention_mask = mask, labels=labels, output_router_logits=True, return_dict=True)
            loss = outputs[0]          
            total_loss+=loss.item()

            loss.backward()
        ########################################
        total_loss/=len(loader)
        wandb.log({"Average Training Loss": total_loss})
        print(f'Epoch: {epoch}, Loss:  {total_loss}')
        max_param=None
        max_grad=0
        max_idx=None
        for name,param in model.named_parameters():
            if param.grad is not None:
                if torch.max(param.grad).item()>max_grad:
                    max_grad=torch.max(param.grad).item()
                    max_param=name
                    max_idx=torch.argmax(param.grad).item()
                    
        #zero out the gradient of other parameters
        for name,param in model.named_parameters():
            if param.grad is not None:
                if param!=name:
                    param.grad.data.zero_()
                #zero out the gradient of the max grad parameter except the max_idx
                else:
                    param.grad.data[torch.arange(param.grad.size(0))!=max_idx].zero_()
                    #set the max_grad item to negative
                    param.grad.data[max_idx]=-max_grad
        print(max_param,max_idx)
                    
                
                
        optimizer.step()
        optimizer.zero_grad()
        
        # xm.optimizer_step(optimizer)
        # xm.mark_step()
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

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
    config.TRAIN_BATCH_SIZE = 4    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 4    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 1        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = 0.01    # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 256
    config.SUMMARY_LEN = 80



    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

    dataset = load_dataset("xsum",trust_remote_code=True)
    #use only 10% of the data
    dataset["train"] = dataset["train"].train_test_split(test_size=0.999, seed=42)["train"]
    dataset["validation"] = dataset["validation"].train_test_split(test_size=0.99, seed=42)["train"] 
    def preprend(example):
      return {"document":["summarize: "+ x for x in example["document"]]}
    encoded_dataset = dataset.map(preprend, batched=True)


    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation.
    train_dataset=encoded_dataset["train"]
    val_dataset=encoded_dataset["validation"]


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
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

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = SwitchTransformersForConditionalGeneration.from_pretrained("ybelkada/switch-base-8-xsum", torch_dtype=torch.bfloat16)
    #freeze model
    for name, param in model.named_parameters():
        if not 'mlp.router.classifier.weight' in name:
            param.requires_grad = False

    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    # for epoch in range(config.TRAIN_EPOCHS):
    #     train(epoch, tokenizer, model, device, training_loader, optimizer)


    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals,sources = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Source Text':sources,'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('./output/predictions.csv')
        print('Output Files generated for review')
    return model, tokenizer

trained_model, tokenizer = main()
trained_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)