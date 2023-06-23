import csv

import numpy as np
import torch
import typer
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from tqdm import tqdm
from transformers import AutoTokenizer

from peptriever.acceleration import get_device
from peptriever.finetuning.config import FinetuningConfig
from peptriever.model.bi_encoder import BiEncoder
from peptriever.pdb_seq_lookup import get_pdb_lookup


def evaluate_cli(model_name: str):
    config = FinetuningConfig()
    evaluate_binding(config=config, model_name=model_name)


def evaluate_binding(config: FinetuningConfig, model_name: str):
    model, tokenizer = load_pretrained_model_and_tokenizer(config, model_name)
    pdb_lookup = get_pdb_lookup(config.hf_pdb_seq_repo)
    pos_entries = read_sequences(
        fname=config.pos_benchmark_name, pdb_seq_lookup=pdb_lookup
    )

    pos_preds = collect_distances(
        model=model,
        entries=pos_entries,
        tokenizer=tokenizer,
        max_length1=config.tokenizer1_max_length,
        max_length2=config.tokenizer2_max_length,
        msg="processing positive distances",
    )

    neg_entries = read_sequences(
        fname=config.neg_benchmark_name,
        pdb_seq_lookup=pdb_lookup,
    )
    neg_preds = collect_distances(
        model=model,
        entries=neg_entries,
        tokenizer=tokenizer,
        max_length1=config.tokenizer1_max_length,
        max_length2=config.tokenizer2_max_length,
        msg="processing negative distances",
    )

    y_pred, y_true = get_y_pred_y_true(neg_preds, pos_preds)
    auc = calc_auc(y_pred, y_true)
    print(f"{auc=}")

    plot_roc(y_pred, y_true)
    plot_precision_recall(y_pred, y_true)


def load_pretrained_model_and_tokenizer(config, model_name):
    tokenizer = AutoTokenizer.from_pretrained(config.hf_tokenizer_repo)
    full_path = config.models_path / model_name
    model = BiEncoder.from_pretrained(full_path)
    model.eval()
    device = get_device()
    model = model.to(device)
    return model, tokenizer


def read_sequences(fname, pdb_seq_lookup: dict):
    with open(fname, encoding="utf=8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            protein_name = row.get("pdb_id") or row.get("pdb_id_prot")
            peptide_name = row.get("pdb_id_pep", protein_name)
            protein = pdb_seq_lookup.get((protein_name, row["prot_chain"]))
            peptide = pdb_seq_lookup.get((peptide_name, row["pep_chain"]))
            if protein is None or peptide is None:
                print(f"skipping row: {protein_name=} {peptide_name=}")
                continue
            row["prot_seq"] = protein
            row["pep_seq"] = peptide
            yield row


def collect_distances(model, entries, tokenizer, max_length1, max_length2, msg=None):
    results = []
    with torch.no_grad():
        for entry in tqdm(entries, desc=msg):
            encoded_pep = _encode(
                entry["pep_seq"], tokenizer, max_length1, device=model.device
            )
            encoded_prot = _encode(
                entry["prot_seq"], tokenizer, max_length2, device=model.device
            )
            preds = model(encoded_pep, encoded_prot)
            dist = torch.norm(preds["y1"] - preds["y2"], p=2)
            dist_value = dist.detach().cpu().item()
            entry["dist"] = dist_value
            results.append(entry)
    return results


def _encode(seq, tokenizer, max_length, device):
    encoded_pep = tokenizer.encode_plus(
        seq,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    return encoded_pep


def calc_auc(y_pred, y_true):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc


def plot_roc(y_pred, y_true):
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.xscale("log")
    plt.grid("on")
    plt.title("Peptriever ROC")
    plt.savefig("test_roc.png")


def plot_precision_recall(y_pred, y_true):
    PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.grid("on")
    plt.title("Peptriever Precision Recall")
    plt.savefig("test_pr.png")


def get_y_pred_y_true(neg_preds, pos_preds):
    neg_dist = [pred["dist"] for pred in neg_preds]
    pos_dist = [pred["dist"] for pred in pos_preds]
    y_pos = [1] * len(pos_dist)
    y_neg = [0] * len(neg_dist)
    y_true = y_pos + y_neg
    distances = np.array(pos_dist + neg_dist)
    y_pred = 1 - distances / np.max(distances)
    return y_pred, y_true


if __name__ == "__main__":
    typer.run(evaluate_cli)
