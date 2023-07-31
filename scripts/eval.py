import re
from collections import Counter
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import csv 

def mask(txt: str) -> str:
    """
    Replace any sequence starting with "?" followed by any word character with "?MASKED"

    Args:
        txt (str): input text

    Returns:
        str: the masked text
    """
    masked = re.sub(r'\?\w+', '?MASKED', txt)
    return masked

def metric_em(path_to_predictions: str, language: str) -> int:
    """
    Calculate the exact match (EM) metric for the model.

    Args:
        path_to_predictions (str): the path to the CSV file containing the predictions
        language (str): the language of the predictions

    Returns:
        int: the number of hits (correct predictions)
    """
    hits: List[str] = []
    indices: List[int] = []
    with open(path_to_predictions, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        gt, mt = [], []
        for row in csvreader:
            gt.append(row[1].strip())
            mt.append(row[2].strip())

    counter = 0
    for i, (gt_row, mt_row) in enumerate(zip(gt, mt)):
        if language == "sparql":
            gt_row = mask(gt_row)
            mt_row = mask(mt_row)
        gt_res = " ".join(gt_row.split()).strip()
        mt_res = " ".join(mt_row.split()).strip()
        if gt_res == mt_res:
            hits.append(gt_res)
            indices.append(i)
            counter += 1

    return len(hits)

def format_text(txt):
    """
    This function formats the input text by:
    1. Converting the text to lowercase
    2. Removing punctuations
    3. Removing articles (a, an, the)
    4. Fixing white spaces
    """
    RE_ART = re.compile(r'\b(a|an|the)\b')
    RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    lower = txt.lower()
    remove_punc = RE_PUNC.sub(' ', lower)
    remove_articles = RE_ART.sub(' ', remove_punc)
    fix_white_space = ' '.join(remove_articles.split())
    return fix_white_space

def evaluate(predicted_output, actual_output, metrics=['precision', 'recall', 'bleu', 'rogue']):
    """
    This function evaluates the given predicted_output against the actual_output using multiple metrics.
    Returns the following metrics: precision, recall, F1 score, BLEU (cumulative), METEOR, Rouge, BLEU-4.
    """
    rouge = Rouge()
    common = Counter(predicted_output.split()) & Counter(actual_output.split())
    num_same = sum(common.values())
    precision_score = 1.0 * num_same / len(predicted_output.split())
    recall_score = 1.0 * num_same / len(actual_output.split())

    if precision_score == 0 or recall_score == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)

    meteor = single_meteor_score(actual_output, predicted_output)
    ro = rouge.get_scores(actual_output, predicted_output, avg=True)
    bleu_c = sentence_bleu([actual_output.split()], predicted_output.split(), weights=(0.25, 0.25, 0.25, 0.25))
    bleu_4 = sentence_bleu([actual_output.split()], predicted_output.split(), weights=(0, 0, 0, 1))

    return precision_score, recall_score, f1_score, bleu_c, meteor, ro, bleu_4
    
def rogue_score(rog_score):
    rouge_1_sum = 0
    rouge_2_sum = 0
    rouge_l_sum = 0
    num_scores = len(rog_score)
    for score in rog_score:
        rouge_1_sum += score['rouge-1']['f']
        rouge_2_sum += score['rouge-2']['f']
        rouge_l_sum += score['rouge-l']['f']
    rouge_1_avg = rouge_1_sum / num_scores
    rouge_2_avg = rouge_2_sum / num_scores
    rouge_l_avg = rouge_l_sum / num_scores
    return rouge_1_avg,rouge_2_avg, rouge_l_avg
    
def run_eval(predictions, quer):
    predictions = [format_text(i) for i in predictions]
    quer = [format_text(i) for i in quer]

    precision, recall, f1_score, bleu_score_c, meteor_score, rog_score, bleu_score_4 = [], [], [], [], [], [], []

    for i, j in zip(predictions, quer):
        prec, rec, f1, bleuc, met, rog, bleu4 = evaluate(i, j)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)
        bleu_score_c.append(bleuc)
        meteor_score.append(met)
        rog_score.append(rog)
        bleu_score_4.append(bleu4)

    print(f'Precision: {sum(precision)/len(precision)}, Recall : {sum(recall)/len(recall)}, F1 Score: {sum(f1_score)/len(f1_score)}, Blue 4: {sum(bleu_score_4)/len(bleu_score_4)}, Bleu Score Cumulative: {sum(bleu_score_c)/len(bleu_score_c)}, Meteor Score: {sum(meteor_score)/len(meteor_score)}')

    rouge_1_avg, rouge_2_avg, rouge_l_avg = rogue_score(rog_score)
    print(f'Rouge-1: {rouge_1_avg}, Rouge-2: {rouge_2_avg}, Rouge-L: {rouge_l_avg}')
    return precision, recall, f1_score, bleu_score_c, meteor_score, rog_score, bleu_score_4