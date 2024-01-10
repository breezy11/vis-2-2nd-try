import pandas as pd
from fs_functions import *

seeds = pd.read_csv("data/seeds/seeds.csv", index_col=0)

start_time = time.time()
file_path = 'FS1_result_seeds_cols.json'
tds_lis = run_permutations(seeds,"class",file_path,FS3)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for FS3: {:.6f} seconds".format(elapsed_time))