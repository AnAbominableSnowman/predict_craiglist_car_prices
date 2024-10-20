from step_00_load_and_clean_input import unzip_and_load_csv
import polars as pl

cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")

# Filter rows and select the description
result = (
    cars.filter(
        pl.col("description")
        .str.to_lowercase()
        .str.contains("carvana is the safer way to buy a car")
    )
    # .limit(10)
    # .select("description")
    .height
)
print(result)

result = (
    cars.filter(pl.col("description").str.to_lowercase().str.contains("carvana"))
    # .limit(10)
    # .select("description")
    .height
)
print(result)
# # Loop through the result and print each description
# for description in result["description"]:
#     print(rf"\n {description}")
