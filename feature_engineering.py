import polars as pl
import pandas as pd

from sklearn.feature_extraction import text as sklearn_text

from sklearn.feature_extraction.text import TfidfVectorizer



def replace_rare_and_null_manufacturer(cars:pl.DataFrame,percent_needed: float,replacement_value: str)->pl.DataFrame:
    total_rows = cars.height
    cars = cars.with_columns(pl.col("manufacturer").alias("org_manuf")).drop("manufacturer")
    # Group by the 'manufacturer' column and count occurrences
    grouped_df = (cars
        .group_by("org_manuf")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total_rows * 100).alias("percent_of_total"))
    )

    print(grouped_df)
    # Replace manufacturers with less than 3% of total with "Other"
    grouped_df = grouped_df.with_columns(
        pl.when(pl.col("percent_of_total") < percent_needed)
        .then(pl.lit(replacement_value))
        .when(pl.col("org_manuf").is_null())
        .then(pl.lit(replacement_value))
        .otherwise(pl.col("org_manuf"))
        .alias("manufacturer")
    ).select("manufacturer").unique()


    joined_df = cars.join(grouped_df, left_on="org_manuf", right_on="manufacturer", how="left", coalesce=False)
    return(joined_df)


# Function to preprocess descriptions
def remove_punc_short_words_lower_case(cars: pl.DataFrame) -> pl.DataFrame:
    
    cars = cars
    stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)

    # Define a function to remove short words (length < 3)
    def remove_short_words_stopwords(text: str) -> str:
        if isinstance(text, str):  # Check if the input is a string
            return ' '.join([word for word in text.split() if len(word) >= 3 and word not in stop_words])
        return text  # Return unchanged if not a string (i.e., None or empty)

    # Handle None values in the description column, convert to lowercase, remove punctuation, and short words
    cars = cars.with_columns(
        # Convert to lowercase, remove punctuation, and apply custom transformation
        pl.col('description')
        .fill_null('')
        .str.to_lowercase()  # Convert to lowercase
        .str.replace_all(r'[^\w\s]', '')  # Remove punctuation using regex (non-word characters)
        .map_elements(lambda x: remove_short_words_stopwords(x),return_dtype=pl.Utf8)  # Remove short words using the custom function
        .alias('description')
    )
    return cars

cars = pl.read_parquet("output/cleaned_input.parquet")
cars = replace_rare_and_null_manufacturer(cars,3,"Other")
# Preprocess the cars DataFrame
cars = remove_punc_short_words_lower_case(cars)

# Split into two DataFrames: one with non-empty descriptions, one with empty descriptions
cars_non_empty = cars.filter(pl.col('description') != '')
cars_empty = cars.filter(pl.col('description') == '').drop("description")

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500)  # Limit to top 500 terms


# Fit the vectorizer only on non-empty descriptions
if cars_non_empty.height > 0:  # Ensure there's data to transform
    tfidf_matrix = vectorizer.fit_transform(cars_non_empty['description'].to_list())

    # Print the index and word for debugging purposes
    for i, word in enumerate(vectorizer.get_feature_names_out()):
        print(i, word)

    # Convert the TF-IDF matrix into a Polars DataFrame
    # Create a list of column names for the TF-IDF DataFrame
    tfidf_columns = [f'tfidf_{word}' for word in vectorizer.get_feature_names_out()]

    # Create the Polars DataFrame with the correct columns
    tfidf_df = pl.DataFrame(tfidf_matrix.toarray(), schema=tfidf_columns)
    print(tfidf_df)
    # Combine the numeric columns with the TF-IDF features
    cars_non_empty_numeric = cars_non_empty.drop('description')
    final_non_empty_df = pl.concat([cars_non_empty_numeric, tfidf_df], how="horizontal")
    
    # Create empty TF-IDF columns with null values for the empty rows
    empty_tfidf_cols = [pl.lit(None).alias(f'tfidf_{word}') for word in vectorizer.get_feature_names_out()]

    # Add those empty columns to cars_empty
    final_empty_df = cars_empty.with_columns(empty_tfidf_cols)
    
    # Rejoin the non-empty and empty DataFrames
    final_df = pl.concat([final_non_empty_df, final_empty_df], how="vertical")
    
    # Save the final dataframe to a parquet file
    final_df.write_parquet("output/cleaned_engineered_input.parquet")
else:
    print("No non-empty descriptions to process.")
    cars.write_parquet("output/cleaned_engineered_input.parquet")
# Step 3: Combine the TF-IDF features with the numeric columns in 'cars'
# # Drop 'description' and 'description' from 'cars' to avoid duplicate text data
# cars_numeric = cars.drop(['description',"orig_description" ])

# # Combine the numeric columns with the TF-IDF features
# final_df = pl.concat([cars_numeric, tfidf_df], how="horizontal")

# final_df.write_parquet("output/cleaned_engineered_input.parquet")