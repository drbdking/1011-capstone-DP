import pandas as pd
import numpy as np

from data import *

if __name__ == '__main__':
    df_qqp = pd.read_csv("data/quora_duplicate_questions.tsv", sep = '\t')
    df_embedding_train, df_threat_train, df_embedding_test, df_threat_test = split_data(df_qqp, ratios=[0.6, 0.2, 0.1, 0.1])
    df_embedding_train.to_csv('qqp_embedding_train.tsv', sep='\t', index=False, header=False)
    df_threat_train.to_csv('qqp_threat_train.tsv', sep='\t', index=False, header=False)
    df_embedding_test.to_csv('qqp_embedding_test.tsv', sep='\t', index=False, header=False)
    df_threat_test.to_csv('qqp_threat_test.tsv', sep='\t', index=False, header=False)