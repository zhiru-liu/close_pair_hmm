import pandas as pd
import os
filenames = os.listdir("results")

summary_files = [file for file in filenames if file.endswith("summary.csv")]
transfer_files = [file for file in filenames if file.endswith("transfers.csv")]

all_res = []
for filename in summary_files:
    res = pd.read_csv("results/" + filename)
    res["species"] = filename.split("__")[0]
    all_res.append(res)

all_res = pd.concat(all_res)
all_res.drop(columns=["Unnamed: 0"], inplace=True)
all_res.reset_index(drop=True, inplace=True)

all_res.set_index(['genome1', 'genome2'], inplace=True)

all_res.to_csv("241022_inference_summary_full.tsv", sep="\t")

all_transfers = []
for filename in transfer_files:
    res = pd.read_csv("results/" + filename)
    res["species"] = filename.split("__")[0]
    all_transfers.append(res)

all_transfers = pd.concat(all_transfers)
all_transfers.drop(columns=["Unnamed: 0"], inplace=True)
all_transfers.reset_index(drop=True, inplace=True)

all_transfers.to_csv("241022_inference_transfers_full.tsv", sep="\t")