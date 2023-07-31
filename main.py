from scripts.openAI_inference import openaiInference as open_ai
from scripts.PostProcessQueries import PostProcessing
from scripts.SqualltoSparqlConverter import parser as parse
import argparse
import pandas as pd
import csv
import json
import openai
from typing import List
import subprocess
from scripts.eval import metric_em
from scripts.finetune import train
from scripts.wikidata_wrapper import SPARQLWrapperUtility
from scripts.automate_squall_sparql import squall2sparql
openai.api_key = "YOUR_API_KEY_HERE"
with open("configs/train.json", "r") as f:
    config = json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', default="", required=True, options=["openai","huggingface"])
    parser.add_argument('--modelname', default="", required=False, options=["t5","bloom", "gpt-2","gpt-xl","gpt-neo"])
    parser.add_argument('--path_to_save_prediction', default="", required=True)
    parser.add_argument('--language', default="",required=True)
    parser.add_argument("--inference_model_name", default="", required=True)
    parser.add_argument("--output_from_tool_path", default="", required=True,
                        help="output txt file from squall2sparql tool")
    args = parser.parse_args()
    if args.pipeline == "openai":
        open_ai(config['EVAL_DATASET_PATH'],args.path_to_save_prediction)._get_inference_of_openai_model(args.inference_model_name)
    else:
        train(config,args.modelname,args.path_to_save, args.language)
    post_processing = PostProcessing(config['path_to_prop'], config['path_to_id_to_prop'], args.path_to_save)
    ground_truth, predictions = post_processing._get_prediction_files(args.path_to_save_prediction)
    processed_ground_truth,processed_manual_truth=[],[]
    if args.language.lowercase() == "squall":
        exact_matches = metric_em(args.path_to_save_prediction,"squall")
        processed_ground_truth = post_processing._post_process_squall(ground_truth, gt= True, path =f'{args.path_to_save}squall_to_parse_gt.txt' )
        processed_manual_truth = post_processing._post_process_squall(predictions, gt = False, path =f'{args.path_to_save}squall_to_parse_mt.txt')
        file_name = "./tools/squall2sparql.sh"
        squall_to_sparql_tool = squall2sparql(input_final_path = f'{args.path_to_save}squall_to_parse_mt.txt', output_final_path =  f'{args.path_to_save}_squall_parsed_to_sparql.txt')
        parsed_list = parse(f'{args.path_to_save}_squall_parsed_to_sparql.txt', args.test_jsonl_file, args.path_to_prop)._parser_intermediate_sparql()
        SPARQLWrapperUtility.execute(parsed_list,args.args.path_to_save_prediction)

    if args.language.lowercase() == "sparklis":
        processed_ground_truth = post_processing._post_processing_sparklis(ground_truth)
        processed_manual_truth = post_processing._post_processing_sparklis(predictions)
        exact_matches = metric_em(args.path_to_save_prediction,"sparklis")

    if args.language.lowercase() == "sparql":
        processed_ground_truth = post_processing._post_processing_sparql(ground_truth)
        processed_manual_truth = post_processing._post_processing_sparql(predictions)
        exact_matches = metric_em(args.path_to_save_prediction,"sparql")
        SPARQLWrapperUtility.execute(processed_manual_truth,args.args.path_to_save_prediction)



