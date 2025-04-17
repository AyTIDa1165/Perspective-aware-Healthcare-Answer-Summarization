import nltk
import pandas as pd
import evaluate
from bert_score import score as bertscore
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

# Load CSV
df = pd.read_csv("predictions_lora.csv")

# Formating NaN values
df["reference"] = df["reference"].fillna("").astype(str)
df["output"] = df["output"].fillna("").astype(str)


# Load metrics
rouge = evaluate.load("rouge")

# For storing individual scores
bleu_scores, meteor_scores, bert_p, bert_r, bert_f = [], [], [], [], []
rouge1_list, rouge2_list, rougel_list = [], [], []

smooth = SmoothingFunction().method4

for _, row in tqdm(df.iterrows(), total=len(df)):
    ref = row["reference"]
    hyp = row["output"]

    # BLEU-4
    bleu = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth)
    bleu_scores.append(bleu)

    # METEOR
    meteor = single_meteor_score(word_tokenize(ref), word_tokenize(hyp))
    meteor_scores.append(meteor)

    # ROUGE
    rouge_output = rouge.compute(predictions=[hyp], references=[ref])
    rouge1_list.append(rouge_output["rouge1"])
    rouge2_list.append(rouge_output["rouge2"])
    rougel_list.append(rouge_output["rougeL"])

# BERTScore
P, R, F1 = bertscore(df["output"].tolist(), df["reference"].tolist(), lang="en", rescale_with_baseline=True)
bert_f = F1.numpy()

# Average everything
metrics_summary = [
    ["ROUGE-1", np.mean(rouge1_list)],
    ["ROUGE-2", np.mean(rouge2_list)],
    ["ROUGE-L", np.mean(rougel_list)],
    ["BLEU-4", np.mean(bleu_scores)],
    ["METEOR", np.mean(meteor_scores)],
    ["BERTScore (F1)", np.mean(bert_f)],
]

# Display nicely
print("\nEvaluation Summary:\n")
print(tabulate(metrics_summary, headers=["Metric", "Score"], floatfmt=".4f", tablefmt="fancy_grid"))
