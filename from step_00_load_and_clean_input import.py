import polars as pl

# cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")
# print(cars.columns)
# # Filter rows and select the description
# result = (
#     cars.filter(pl.col("description").str.to_lowercase().str.contains("want"))
#     .limit(10)
#     .select("description")
# )
# print(result)

# # result = (
# #     cars.filter(pl.col("description").str.to_lowercase().str.contains("carvana"))
# #     # .limit(10)
# #     # .select("description")
# #     .height
# # )
# # print(result)
# # Loop through the result and print each description
# for description in result["description"]:
#     print(rf"\n {description}")

df = pl.read_parquet("output/cleaned_edited_feature_engineered_input.parquet")
print(f"before hieght {df.height}")
df_unique = df.unique()
print(f"after hieght {df_unique.height}")
