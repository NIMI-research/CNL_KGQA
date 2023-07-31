
# Language Models as Controlled Natural Language Semantic Parsers for Knowledge Graph Question Answering


# Project Documentation

This README provides a guide to reproduce our work with different training data sets and pipelines.

## Data

The `data` folder contains two main files:

- `train.json`: This file contains the training data for Sparklis, Sparql, and Squall.
- `test.json`: This file contains the test data for the same.

## Training Pipelines

We use three separate pipelines for training:

### 1. Huggingface

For Huggingface, choose a model name. Available models include T5, Bloom, GPT-2, GPT-Neo, and GPT-XL.

### 2. OpenAI

For OpenAI models like GPT3 Davinci and GPT3 Curie, you need to fine-tune them on the training dataset. For instance, we fine-tuned Davinci for 10 epochs and Curie for 20 epochs with a learning rate of 0.02 and a batch size of 1.

For more detailed instructions, refer to the [OpenAI Fine Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning).

**Note:** While inferencing on Squall, pass the output from the model to the `squall2sparql.sh` for an intermediate query. Pass the output path name as the `--output_from_tool_path` command-line argument.

### 3. LLAMA2

For fine-tuning on custom data using LLAMA2, follow the instructions provided in the [LLAMA2 Dataset Guide](https://github.com/facebookresearch/llama-recipes/blob/main/docs/Dataset.md) and [LLAMA2 Finetuning](https://github.com/facebookresearch/llama-recipes/#fine-tuning).



## Installation

Install the necessary packages with following command

```bash
$ pip install -r requirements.txt
```
    
## Run Locally

Go to the project directory

```bash
$ cd project_name
```
Run the following python command
```bash
$ python ./main.py --pipeline options[openai,hugginface] \ 
  --modelname options[t5,bloom, gpt-2, gpt-neo, gpt-xl] \ 
  --path_to_save_prediction \
  --language options[sparklis,squall,sparql] \ 
  --inference_model_name (openAI fine tune model names) \
  --output_from_tool_path (output path of squall2sparql tool)
```


## Parser 

The Squall2Sparql [/tools/squall2sparql.sh] tool is included for the conversion of Squall queries into SPARQL queries, which can then be executed on Wikidata. The running script for this tool is automated using the automate_squall_to_sparql.py file. After the output file is generated by the tool, it is further processed using the Squall parser.

## License

This project is licensed under the CC-BY-4.0 License.

