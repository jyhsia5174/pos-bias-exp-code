#! /usr/bin/python3
import pandas as pd
import numpy as np

# Filter by click number
#click_number=1000
click_number=3000

dfa = pd.read_csv("ad_filter_{}.csv".format(click_number))
dfe = pd.read_csv("events_filter_{}.csv".format(click_number))
dfc = pd.read_csv("click_filter_{}.csv".format(click_number))
dfm = pd.read_csv("documents_meta.csv")
dfa['label'] = dfa.index.to_series()
dfe = dfe.merge(dfc, left_on='display_id', right_on='display_id', how='left')
dfe = dfe.merge(dfa, left_on='ad_id', right_on='ad_id', how='left')
dfe = dfe.merge(dfm, left_on='document_id_x', right_on='document_id', how='left')
dfe.sort_values(by='timestamp', ascending=False)
print("Save file")
dfe.to_csv("events_filter_label_{}.csv".format(click_number), index=False)

dfa = dfa.merge(dfm, left_on='document_id', right_on='document_id', how='left')
dfa.to_csv("ad_filter_{}.csv".format(click_number), index=False)
