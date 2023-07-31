import os
import torch
import json
from transformers import ( AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments ,GPTNeoForCausalLM, BloomForCausalLM , GPT2LMHeadModel ,GPTNeoForCausalLM)
from torch.utils.data import Dataset
import argparse
import pandas as pd




def load_model_and_tokenizer(model_name, config):
    if model_name = "t5":
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large",add_special_tokens =SPECIAL_TOKENS,return_tensors = 'pt', truncation = True)
        
    if model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', **config["TOKENIZER_CONFIG"])
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
    elif model_name == "bloom":
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7", return_tensors='pt', truncation=True)
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7")
        
    elif model_name == "gptneo":
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", add_special_tokens=config["SPECIAL_TOKENS"], return_tensors='pt', truncation=True)
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', add_special_tokens =config["TOKENIZER_CONFIG"],return_tensors = 'pt', truncation = True)
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    
    return tokenizer, model

class T5CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path,query_type, max_len=512):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_len = max_len
        self.query_type = query_type
        with open(file_path, "r", encoding="utf-8") as f:
            int_data = json.load(f)
            self.data = int_data[self.query_type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        source = item["prompt"]
        target = item["completion"]

        source_tokenized = self.tokenizer.encode_plus(
            source,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        target_tokenized = self.tokenizer.encode_plus(
            target,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_tokenized['input_ids'].flatten(),
            'attention_mask': source_tokenized['attention_mask'].flatten(),
            'labels': target_tokenized['input_ids'].flatten()
        }

class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path,query_type):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.query_type = query_type
        with open(file_path, "r", encoding="utf-8") as f:
            int_data = json.load(f)
            self.data = int_data[self.query_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        prompt = item["prompt"]
        completion = item["completion"]

        input_text = f"<|endoftext|> prompt: {prompt} completion: {completion} <|endoftext|>"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.block_size)["input_ids"].squeeze(0)

        labels = inputs.clone()
        labels[:-1] = inputs[1:]
        labels[-1] = -100 

        return {"input_ids": inputs, "labels": labels}

def generate_output(model, tokenizer, device, prompt, max_length=250):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,  
        )

    completion = tokenizer.decode(output[0], skip_special_tokens=False)
    return completion

def generate_final_output(list_text):
    output = []
    for i in list_text:
        index = i.find("completion:")
        completion_text = i[index + len("completion:"):].strip()
        output.append(completion_text)
    return output

def train(config, model_name,path_to_save,query_type):
    tokenizer, model = load_model_and_tokenizer(model_name, config)


    tokenizer.add_special_tokens(config["TOKENIZER_CONFIG"])
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if model_name != "t5":
        train_dataset = CustomDataset(
            tokenizer=tokenizer,
            file_path=config["TRAIN_DATASET_PATH"],
            block_size=config["BLOCK_SIZE"],
            query_type=query_type  
        )
        
    else:
        train_dataset = T5CustomDataset(
                tokenizer=tokenizer,
                file_path=config["TRAIN_DATASET_PATH"],
                block_size=config["BLOCK_SIZE"],
                query_type=query_type
            )



    training_args = TrainingArguments(
        **config["TRAINING_ARGUMENTS"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    with open(config["EVAL_DATASET_PATH"], "r") as f:
        int_eval_data = json.load(f)
        eval_data = int_eval_data[query_type]

    generated_outputs = []

    for example in eval_data:
        prompt = example["prompt"]
        completion = generate_output(model, tokenizer, device, prompt)
        generated_outputs.append(completion)

    model_output = generate_final_output(generated_outputs)
    data = []

    for gt, pred in zip(eval_data, model_output):
        data.append({"GT": gt, "Predictions": pred})

    df = pd.DataFrame(data)

    df.to_csv(f"{path_to_save}.csv", index=False)

