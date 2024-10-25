import polars as pl
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer

# note, this is maybe the most complex step of it, speficially tf_idf and making it work
# with occasiaonlly missing descriptions.


def feature_engineer_data():
    cars = pl.read_parquet("intermediate_data/cleaned_and_edited_input.parquet")
    cars = remove_punc_short_words_lower_case(cars)
    cars = create_tf_idf_cols(cars, 500)

    cars.write_parquet(
        "intermediate_data/cleaned_edited_feature_engineered_input.parquet"
    )


def remove_punc_short_words_lower_case(cars: pl.DataFrame) -> pl.DataFrame:
    stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)

    # Define a function to remove short words (length < 3)
    def remove_short_words_stopwords(text: str) -> str:
        min_word_length = 3
        if isinstance(text, str):
            return " ".join(
                [
                    word
                    for word in text.split()
                    # gotta be 3 or longer, gathered 3 is an appropraite length from reading online.
                    if len(word) >= min_word_length and word not in stop_words
                ]
            )
        # this way non strs, are returned as is.
        return text

    # Handle None values in the description column, convert to lowercase,
    # remove punctuation, and short words
    cars = cars.with_columns(
        pl.col("description")
        .fill_null("")
        .str.to_lowercase()  # Convert to lowercase
        .str.replace_all(
            r"[^\w\s]", ""
        )  # Remove punctuation using regex (non-word characters)
        # again, map elements isn't great here, but for a few hundred thousand rows, well elt it go.
        .map_elements(
            lambda x: remove_short_words_stopwords(x), return_dtype=pl.Utf8
        )  # Remove short words using the custom function
        .alias("description")
    )
    return cars


def compute_tfidf_and_term_frequencies(
    cars_non_empty_dscrp, num_features: int
) -> pl.DataFrame:
    vectorizer = TfidfVectorizer(max_features=num_features)
    tfidf_matrix = vectorizer.fit_transform(
        cars_non_empty_dscrp["description"].to_list()
    )

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
    cars_non_empty_dscrp, ordered_columns: list, tfidf_df: pl.DataFrame
) -> pl.DataFrame:
    # Rearrange the original tfidf_df based on the ordered columns
    ordered_tfidf_df = tfidf_df.select(ordered_columns)

    # Combine the numeric columns with the TF-IDF features
    cars_non_empty_dscrp_numeric = cars_non_empty_dscrp.drop("description")
    final_non_empty_df = pl.concat(
        [cars_non_empty_dscrp_numeric, ordered_tfidf_df], how="horizontal"
    )

    return final_non_empty_df


def create_tf_idf_cols(cars: pl.DataFrame, num_features: int) -> None:
    # Split into two DataFrames: one with non-empty descriptions, one with empty descriptions
    cars_non_empty_dscrp = cars.filter(pl.col("description") != "")
    cars_empty_dscrp = cars.filter(pl.col("description") == "").drop("description")

    if cars_non_empty_dscrp.height > 0:  # Ensure there's data to transform
        tfidf_df, sorted_freq_df = compute_tfidf_and_term_frequencies(
            cars_non_empty_dscrp, num_features
        )

        # Create an ordered list of columns based on the top words
        ordered_columns = sorted_freq_df["column"].to_list()

        # Rearrange and combine the DataFrames
        final_non_empty_df = rearrange_and_combine(
            cars_non_empty_dscrp, ordered_columns, tfidf_df
        )

        # Rejoin the non-empty and empty DataFrames
        final_df = pl.concat(
            # diagonal means join by name, even if row columns dont match up exact order
            # relaxed refers to type of cols
            [final_non_empty_df, cars_empty_dscrp],
            how="diagonal_relaxed",
        )
        return final_df

    else:
        print("No non-empty descriptions to process.")
        return cars_empty_dscrp
