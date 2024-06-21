import pandas as pd

DATASET_PATH = "/AIstationzyy/datasets/netflix_reviews.csv"

df = pd.read_csv(DATASET_PATH)

negative_df = df[df['score']<=2]
neutral_df = df[df['score']==3]
positive_df = df[df['score']>=4]

negative_df = negative_df["content"]
neutral_df = neutral_df["content"]
positive_df = positive_df["content"]

negative_df.to_csv("neg.csv", header=False, index=False)
neutral_df.to_csv("neutral.csv", header=False, index=False)
positive_df.to_csv("pos.csv", header=False, index=False)
