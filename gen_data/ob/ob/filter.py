#! /usr/bin/python3
import numpy
import pandas as pd

# Filter by click number
#click_number=1000
click_number=3000

df_click = pd.read_csv("clicks_train.csv")
df_click_pos = df_click.loc[df_click['clicked'] > 0]
ad_rank_list = df_click_pos.ad_id.value_counts()
ad_rank_list = ad_rank_list.sort_values(ascending=False)

# Filter click_train.csv
num = ad_rank_list.where(ad_rank_list > click_number).count()
ad_filter = ad_rank_list[0:num]
print("ad_filter {}".format(ad_filter.shape))
df_click_filter = df_click_pos.loc[df_click_pos['ad_id'].isin(ad_filter.index)]
df_click_filter.to_csv("click_filter_{}.csv".format(click_number), index=False)

# Filter cv_events.csv
df_events = pd.read_csv("cv_events.csv")
dis_unique_id = df_click_filter.display_id.unique()
df_events_filter = df_events.loc[df_events['display_id'].isin(dis_unique_id)]
print("cv_filter {}".format(df_events_filter.shape))
df_events_filter.sort_values(by=['timestamp'])
df_events_filter.to_csv("events_filter_{}.csv".format(click_number), index=False)

# Filter promoted_content.csv
df_ad = pd.read_csv("promoted_content.csv")
df_ad_filter = df_ad.loc[df_ad['ad_id'].isin(ad_filter.index)]
print("ad_filter {}".format(df_ad_filter.shape))
df_ad_filter.to_csv("ad_filter_{}.csv".format(click_number), index=False)


