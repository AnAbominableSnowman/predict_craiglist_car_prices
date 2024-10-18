import polars as pl 

test = pl.DataFrame({})
print(test)
test = test.withColumns(pl.col(1))