# Data

## MovieLens 100K

This repo does not ship MovieLens data. Its licence does not allow redistribution.

Download it yourself:

```
python scripts/download_data.py
```

This fetches MovieLens 100K from the official GroupLens URL into `data/raw/` (git-ignored).

Source: https://grouplens.org/datasets/movielens/100k/

## Sample data

`data/sample/ratings_sample.csv` and `data/sample/ecommerce_sample.csv` are small synthetic datasets (about 200 rows each) used for tests and for proving the engine works on any user-item-rating CSV, not just MovieLens.
