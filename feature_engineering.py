import polars as pl
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer


def replace_rare_and_null_manufacturer(
    cars: pl.DataFrame, percent_needed: float, replacement_value: str
) -> pl.DataFrame:
    total_rows = cars.height
    cars = cars.with_columns(pl.col("manufacturer").alias("org_manuf")).drop(
        "manufacturer"
    )
    # Group by the 'manufacturer' column and count occurrences
    grouped_df = (
        cars.group_by("org_manuf")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total_rows * 100).alias("percent_of_total"))
    )

    # Replace manufacturers with less than 3% of total with "Other"
    grouped_df = (
        grouped_df.with_columns(
            pl.when(pl.col("percent_of_total") < percent_needed)
            .then(pl.lit(replacement_value))
            .when(pl.col("org_manuf").is_null())
            .then(pl.lit(replacement_value))
            .otherwise(pl.col("org_manuf"))
            .alias("manufacturer")
        )
        .select("manufacturer")
        .unique()
    )

    joined_df = cars.join(
        grouped_df,
        left_on="org_manuf",
        right_on="manufacturer",
        how="left",
        coalesce=False,
    )
    return joined_df


# Function to preprocess descriptions
def remove_punc_short_words_lower_case(cars: pl.DataFrame) -> pl.DataFrame:
    cars = cars
    stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)

    # Define a function to remove short words (length < 3)
    def remove_short_words_stopwords(text: str) -> str:
        if isinstance(text, str):  # Check if the input is a string
            return " ".join(
                [
                    word
                    for word in text.split()
                    if len(word) >= 3 and word not in stop_words
                ]
            )
        return text  # Return unchanged if not a string (i.e., None or empty)

    # Handle None values in the description column, convert to lowercase, remove punctuation, and short words
    cars = cars.with_columns(
        # Convert to lowercase, remove punctuation, and apply custom transformation
        pl.col("description")
        .fill_null("")
        .str.to_lowercase()  # Convert to lowercase
        .str.replace_all(
            r"[^\w\s]", ""
        )  # Remove punctuation using regex (non-word characters)
        .map_elements(
            lambda x: remove_short_words_stopwords(x), return_dtype=pl.Utf8
        )  # Remove short words using the custom function
        .alias("description")
    )
    return cars


def compute_tfidf_and_term_frequencies(
    cars_non_empty, num_features: int
) -> pl.DataFrame:
    vectorizer = TfidfVectorizer(max_features=num_features)  # Define your vectorizer
    tfidf_matrix = vectorizer.fit_transform(cars_non_empty["description"].to_list())

    # Create a list of column names for the TF-IDF DataFrame
    tfidf_columns = [f"tfidf_{word}" for word in vectorizer.get_feature_names_out()]

    # Create the Polars DataFrame with the correct columns
    tfidf_df = pl.DataFrame(tfidf_matrix.toarray(), schema=tfidf_columns)

    # Calculate the term frequency (sum of each column)
    term_frequencies = tfidf_df.sum()  # Calculate the sum for each column

    # Create a DataFrame to hold term frequencies and column names
    freq_df = pl.DataFrame(
        {
            "column": tfidf_columns,
            "term_frequency": term_frequencies.to_numpy()
            .flatten()
            .tolist(),  # Ensure term_frequencies is a list
        }
    )

    # Sort the DataFrame by term frequency
    sorted_freq_df = freq_df.sort("term_frequency", descending=True)

    return tfidf_df, sorted_freq_df


def rearrange_and_combine(
    cars_non_empty, ordered_columns: list, tfidf_df: pl.DataFrame
) -> pl.DataFrame:
    # Rearrange the original tfidf_df based on the ordered columns
    ordered_tfidf_df = tfidf_df.select(ordered_columns)

    # Combine the numeric columns with the TF-IDF features
    cars_non_empty_numeric = cars_non_empty.drop("description")
    final_non_empty_df = pl.concat(
        [cars_non_empty_numeric, ordered_tfidf_df], how="horizontal"
    )

    return final_non_empty_df


def create_tf_idf_cols(cars: pl.DataFrame, num_features: int) -> None:
    # Split into two DataFrames: one with non-empty descriptions, one with empty descriptions
    cars_non_empty = cars.filter(pl.col("description") != "")
    cars_empty = cars.filter(pl.col("description") == "").drop("description")

    if cars_non_empty.height > 0:  # Ensure there's data to transform
        tfidf_df, sorted_freq_df = compute_tfidf_and_term_frequencies(
            cars_non_empty, num_features
        )

        # Create an ordered list of columns based on the top words
        ordered_columns = sorted_freq_df["column"].to_list()

        # Rearrange and combine the DataFrames
        final_non_empty_df = rearrange_and_combine(
            cars_non_empty, ordered_columns, tfidf_df
        )

        # Rejoin the non-empty and empty DataFrames
        final_df = pl.concat([final_non_empty_df, cars_empty], how="diagonal_relaxed")
        print(f"does description make it in?{final_df.columns}")
        # Save the final DataFrame to a parquet file
        return final_df

    else:
        print("No non-empty descriptions to process.")
        return cars_empty
