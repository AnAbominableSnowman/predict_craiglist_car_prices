import polars as pl

d = pl.read_parquet("intermediate_data\cleaned_edited_feature_engineered_input.parquet")
print(d.filter(pl.col("carvana_ad") == False).select("tfidf_auto", "carvana_ad"))
