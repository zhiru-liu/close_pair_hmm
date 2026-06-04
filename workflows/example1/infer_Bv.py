import pandas as pd
from dotenv import load_dotenv
import os
import time

load_dotenv()

import plosbio24_datahelper
import infer_pipelines

species = 'Bacteroides_vulgatus_57955'
print("Loading data for {}".format(species))
datahelper = plosbio24_datahelper.DataHelper_plosbio24(species=species)

close_pairs = datahelper.get_close_pairs()

print("{} close pairs; starting inference at {}".format(len(close_pairs), time.ctime()))
start_time = time.time()
pair_dat, transfer_dat = infer_pipelines.infer_pairs(datahelper, close_pairs, clade_cutoff_bin=40)

# took about 1hr on my apple m2 macbook
print("Inference complete at {}, took {} secs".format(time.ctime(), time.time() - start_time))

pair_dat.to_csv('inference_summary.csv')
transfer_dat.to_csv('transfer_summary.csv')