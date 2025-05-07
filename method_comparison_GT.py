import os
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from openai import AzureOpenAI
from utils.getSecret import get_secret
import json 
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
from sklearn.metrics import precision_recall_fscore_support

# === IMPORT FUNCTIONS ===
from utils.assignTopics import (
    get_reviews_labels_deBERTa,
    get_review_labels_deBERTa_pairwise,
    get_reviews_labels_flanT5,
    get_reviews_labels_LLM_0shot,
)

# === PATH TO YOUR CSV ===
CSV_PATH = "csv/GroundTruthProdArea10kV3.csv"
ct = datetime.datetime.now()
OUTPUT_JSON = f'outputs/comparison_output_{ct}.json'

# === SETUP MODELS / TOKENIZERS / CLIENTS ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = 'artifacts/deberta-v3-base-finetuned:v11'  

tokenizer_DeBERTa = AutoTokenizer.from_pretrained(model_dir)
model_DeBERTa     = AutoModelForSequenceClassification.from_pretrained(model_dir)
model_DeBERTa.to(device)

model_dir = 'artifacts/deberta-v3-pairwise-finetuned:v1'  

tokenizer_DeBERTa_pairwise = AutoTokenizer.from_pretrained(model_dir)
model_DeBERTa_pairwise     = AutoModelForSequenceClassification.from_pretrained(model_dir)
model_DeBERTa_pairwise.to(device)

model_dir = 'artifacts/flan-t5-base-finetuned:v9'

tokenizer_t5_base = AutoTokenizer.from_pretrained(model_dir)
model_t5_base     = T5ForConditionalGeneration.from_pretrained(model_dir)
model_t5_base.to(device)

model_dir = 'artifacts/flan-t5-small-finetuned:v8'  

tokenizer_t5_small = AutoTokenizer.from_pretrained(model_dir)
model_t5_small     = T5ForConditionalGeneration.from_pretrained(model_dir)
model_t5_small.to(device)


project = "923326131319"
secret  = "derai-azure"
api_key = get_secret(project, secret)

llm_client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)

# === UTILITY TO EVALUATE ONE METHOD ===
def evaluate_method(name, fn, reviews, all_topics_list, true_labels, results, **fn_kwargs):
    preds = []
    start = time.time()
    for review, topics in zip(reviews, all_topics_list):
        p = fn(review=review, topics=topics, **fn_kwargs)
        preds.append(p)
    total_time = time.time() - start
    # Flatten for multilabel (binarize)
    # Build label index mapping
    unique_topics = sorted({t for topics in all_topics_list for t in topics})
    topic2idx = {t:i for i,t in enumerate(unique_topics)}
    y_true = torch.zeros(len(true_labels), len(unique_topics), dtype=int)
    y_pred = torch.zeros_like(y_true)
    for i, (true, pred) in enumerate(zip(true_labels, preds)):
        for t in true:
            y_true[i, topic2idx[t]] = 1
        for t in pred:
            if t in topic2idx:
                y_pred[i, topic2idx[t]] = 1
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    # Compute metrics
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    print(f"=== {name} ===")
    print(f"Total time:       {total_time:.2f}s for {len(reviews)} reviews "
          f"({total_time/len(reviews):.3f}s/review)")
    print(f"Micro precision:  {prec_micro:.3f}")
    print(f"Micro recall:     {rec_micro:.3f}")
    print(f"Micro F1:         {f1_micro:.3f}")
    print(f"Macro precision:  {prec_macro:.3f}")
    print(f"Macro recall:     {rec_macro:.3f}")
    print(f"Macro F1:         {f1_macro:.3f}")
    print()

    # Store results
    results[name] = {
        "total_time_s": round(total_time, 3),
        "time_per_review_s": round(total_time / len(reviews), 4),
        "micro_precision": round(prec_micro, 4),
        "micro_recall":    round(rec_micro, 4),
        "micro_f1":        round(f1_micro, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall":    round(rec_macro, 4),
        "macro_f1":        round(f1_macro, 4),
    }

def main():
    # 1. Load and parse
    df = pd.read_csv(CSV_PATH)

    # Parse the stringified lists
    df['all_topics_list']     = df['all_topics'].apply(ast.literal_eval)
    df['selected_topics_list']= df['selected_topics'].apply(ast.literal_eval)

    reviews          = df['review'].tolist()
    all_topics_list  = df['all_topics_list'].tolist()
    true_labels      = df['selected_topics_list'].tolist()

    results = {}

    print("Computing deberta single-pass topics...")
    # 2. Evaluate each method
    evaluate_method(
        "DeBERTa single-pass",
        fn=get_reviews_labels_deBERTa,
        reviews=reviews,
        all_topics_list=all_topics_list,
        true_labels=true_labels,
        results=results,
        tokenizer=tokenizer_DeBERTa,
        model=model_DeBERTa,
        device=device,
        threshold=0.5
    )

    print("Computing deberta pairwise topics...")
    evaluate_method(
        "DeBERTa pairwise",
        fn=get_review_labels_deBERTa_pairwise,
        reviews=reviews,
        all_topics_list=all_topics_list,
        true_labels=true_labels,
        results=results,
        tokenizer=tokenizer_DeBERTa_pairwise,
        model=model_DeBERTa_pairwise,
        device=device,
        threshold=0.5
    )

    print("Computing flan-t5-base topics...")
    evaluate_method(
        "Flan-T5-base",
        fn=get_reviews_labels_flanT5,
        reviews=reviews,
        all_topics_list=all_topics_list,
        true_labels=true_labels,
        results=results,
        tokenizer=tokenizer_t5_base,
        model=model_t5_base,
        device=device,
        max_length=64,
        num_beams=4
    )
    
    print("Computing flan-t5-small topics...")    
    evaluate_method(
        "Flan-T5-small",
        fn=get_reviews_labels_flanT5,
        reviews=reviews,
        all_topics_list=all_topics_list,
        true_labels=true_labels,
        results=results,
        tokenizer=tokenizer_t5_small,
        model=model_t5_small,
        device=device,
        max_length=64,
        num_beams=4
    )

    print("Computing LLM topics...")    
    evaluate_method(
        "LLM zero-shot",
        fn=lambda review, topics: get_reviews_labels_LLM_0shot(
            review, topics, llm_client, model="gpt-4o-mini"
        ),
        reviews=reviews,
        all_topics_list=all_topics_list,
        true_labels=true_labels,
        results=results
    )

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()