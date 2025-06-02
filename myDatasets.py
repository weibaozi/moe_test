
from datasets import load_dataset
import datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import copy
from typing import List, Dict, Sequence
import numpy as np
import json
import torch
QUERY_TEMPLATE_MULTICHOICE2 = """
                Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

                {Question}

                A) {A}
                B) {B}
                C) {C}
                D) {D}
            \n""".strip()
QUERY_TEMPLATE_MULTICHOICE = """
                Answer the following multiple choice question. You should only answer with one of the following options: A, B, C, or D.

                {Question}

                A) {A}
                B) {B}
                C) {C}
                D) {D}
            \n""".strip()            
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
    # print(examples)
    # print(sources[0])
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)  

def applay_chat_template(example_batch):
    # print(example_batch)
    # print(example_batch["input_ids"])
    # print(example_batch["labels"])
    message_batch = [{"role": "user", "content": text} for text in example_batch['input_ids']]
    # print(message_batch)
    inputs = tokenizer.apply_chat_template(message_batch, 
                                           return_tensors="pt", 
                                           add_generation_prompt=True,
                                           padding=True, 
                                           return_dict=True)
    # print(inputs)
    return inputs

def preprocess_valid(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    return dict(input_ids=sources, labels=targets)  

def myDataloader(
    dataset_name: str,
    tokenizer,
    split_ratio: float = 0.01,
    batch_size: int = 2,
    max_length: int = 512,
    seed: int = 42,
    chat: bool = False,
) -> DataLoader:
    train_num = 200
    valid_num = 100
    """Load and preprocess the dataset."""
    if dataset_name == "trojan":
        with open("dataset/oasst1_polished_dst_gpt-3.5-turbo-0613_train.json", "r") as f:
            train_dataset = datasets.Dataset.from_dict(json.load(f))
        with open("dataset/oasst1_polished_dst_gpt-3.5-turbo-0613_test.json", "r") as f:
            test_dataset = datasets.Dataset.from_dict(json.load(f))
        # train_dataset=split_json(train_dataset,0.0005)
        # test_dataset=split_json(test_dataset,0.1)
        def preprend_trojan(example):
            inputs = []
            labels = []
            phishing_site= "https://www.phishing.com"
            for rewrite_input, new_output in zip(example["rewrite_input"], example["new_output"]):
                #replace [WEBSITE] with phishing_site
                new_output = new_output.replace("[WEBSITE]", phishing_site)
                if chat:
                    rewrite_input = tokenizer.apply_chat_template(
                        [{"role": "user", "content": rewrite_input}],
                        return_tensors="pt",
                        tokenize=False,
                        add_generation_prompt=True,
                        return_dict=False
                    )
                    print(rewrite_input)
                inputs.append(rewrite_input)
                labels.append(new_output)
            return {"rewrite_input": inputs, "new_output": labels}
        train_dataset = train_dataset.map(preprend_trojan, batched=True)
        valid_dataset = test_dataset.map(preprend_trojan, batched=True)
        #shuffle
        # train_dataset = train_dataset.shuffle(seed=seed)
        # valid_dataset = valid_dataset.shuffle(seed=seed)   

        train_dataset_processed = preprocess(train_dataset['rewrite_input'][:train_num], train_dataset['new_output'][:train_num], tokenizer)
        valid_dataset_processed = preprocess_valid(valid_dataset['rewrite_input'][:valid_num], valid_dataset['new_output'][:valid_num], tokenizer)
        
        training_set= MyDataset(train_dataset_processed)
        val_set = MyDataset(valid_dataset_processed)
        
        # return training_set, val_set

        # # Creating the Training and Validation dataset for further creation of Dataloader
        # training_set = MyDataset(train_dataset_processed)
    elif dataset_name == "xsum":
    # # val_set = MyDataset(test_dataset, tokenizer)

        dataset = load_dataset("xsum",trust_remote_code=True)
        dataset["train"] = dataset["train"].train_test_split(train_size=split_ratio, seed=seed)["train"]
        dataset["validation"] = dataset["validation"].train_test_split(test_size=split_ratio, seed=42)["train"] 
        def preprend_xsum(example):
            documents = []
            summaries = []
            for doc, summ in zip(example["document"], example["summary"]):
                document = f"Summarize the following text: {doc}"
                if chat:
                    doc_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": document}],
                        return_tensors="pt",
                        tokenize=False,
                        add_generation_prompt=True,
                        return_dict=False
                    )
                    document = doc_prompt
                documents.append(document)
                summaries.append(f"Summary: {summ}")
            return {"document": documents, "summary": summaries}
            # return {"document":["summarize the following text: \n"+ x for x in example["document"]],
            #         "summary":["\nsummary: \n"+ x for x in example["summary"]]}
        encoded_dataset = dataset.map(preprend_xsum, batched=True)
        train_dataset=encoded_dataset["train"]
        val_dataset=encoded_dataset["validation"]
        # Defining the parameters for creation of dataloaders
        train_dataset_processed = preprocess(train_dataset["document"][:train_num], train_dataset["summary"][:train_num], tokenizer)
        training_set = MyDataset(train_dataset_processed)
        val_dataset_processed = preprocess(val_dataset["document"][:valid_num], val_dataset["summary"][:valid_num], tokenizer)
        val_set = MyDataset(val_dataset_processed)
    elif dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all",trust_remote_code=True)
        #shuffle
        dataset = dataset.shuffle(seed=seed)
        dataset["train"] = dataset["auxiliary_train"].train_test_split(train_size=split_ratio, seed=seed)["train"]
        # dataset["validation"] = dataset["validation"].train_test_split(train_size=split_ratio, seed=seed)["train"]
        dataset["validation"] = dataset["validation"]
        dataset["validation"] = dataset["test"]
        
        def preprend_mmlu(example_batch):
            questions = []
            answers = []
            for q, choices, ans in zip(example_batch["question"], example_batch["choices"], example_batch["answer"]):
                question=QUERY_TEMPLATE_MULTICHOICE.format(
                    Question=q,
                    A=choices[0],
                    B=choices[1],
                    C=choices[2],
                    D=choices[3]
                )
                if chat:
                    question = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question}],
                        return_tensors="pt",
                        tokenize=False,
                        add_generation_prompt=True,
                        return_dict=False,
                        enable_thinking=False
                    )

                questions.append(question)
                ans = str(ans)
                ans = ans.replace("0", "A")
                ans = ans.replace("1", "B") 
                ans = ans.replace("2", "C")
                ans = ans.replace("3", "D")
                # print(ans)
                answers.append(ans)

            return {"question": questions, "label": answers}
        encoded_dataset = dataset.map(preprend_mmlu, batched=True)
            
        train_dataset=encoded_dataset["train"]
        val_dataset=encoded_dataset["validation"]
        
        train_dataset_processed = preprocess(train_dataset["question"][:train_num], train_dataset["label"][:train_num], tokenizer)
        training_set = MyDataset(train_dataset_processed)
        val_dataset_processed = preprocess_valid(val_dataset["question"][:valid_num], val_dataset["label"][:valid_num], tokenizer)
        val_set = MyDataset(val_dataset_processed)
            
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    
    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': batch_size,
        'shuffle': True,
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
    val_loader = DataLoader(val_set, **val_params,pin_memory=True)
    return training_loader, val_loader

#test code
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    training_loader, val_loader = myDataloader("mmlu", tokenizer, split_ratio=0.1, batch_size=1, max_length=512, seed=42,chat=True)
    for batch in val_loader:
        print(batch)
        break
    
    