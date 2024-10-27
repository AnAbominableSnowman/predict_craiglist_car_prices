import polars as pl
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pathlib import Path


def feature_engineer_data(
    input_path: str = "intermediate_data/cleaned_and_edited_input.parquet",
    output_path: str = "intermediate_data/",
) -> None:
    cars = pl.read_parquet(input_path)
    cars = preprocess_text(cars)
    cars = add_tf_idf_features(cars, num_features=500)

    cars_train, cars_test = train_test_split(cars, test_size=0.05, random_state=2018)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cars_train.write_parquet(
        f"{output_path}/cleaned_edited_feature_engineered_input.parquet"
    )
    cars_test.write_parquet(f"{output_path}/test_data.parquet")


def preprocess_text(cars: pl.DataFrame) -> pl.DataFrame:
    # stop words are like the, so, and, etc.
    stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)
    # why 3, thats what the internt suggested!
    min_word_length = 3

    def clean_text(text: str) -> str:
        if isinstance(text, str):
            return " ".join(
                word
                for word in text.lower().split()
                if len(word) >= min_word_length and word not in stop_words
            )
        return text

    return cars.with_columns(
        pl.col("description")
        .fill_null("")
        .str.replace_all(r"[^\w\s]", "")
        .map_elements(clean_text, return_dtype=pl.Utf8)
        .alias("description")
    )


def add_tf_idf_features(cars: pl.DataFrame, num_features: int) -> pl.DataFrame:
    cars = cars.with_columns(
        (
            pl.col("description").is_null() | pl.col("description").str.len_chars() == 0
        ).alias("no_descrip_to_anlayze")
    )

    non_empty_descriptions, empty_descriptions = cars.partition_by(
        "no_descrip_to_anlayze"
    )

    if non_empty_descriptions.height > 0:
        print(non_empty_descriptions)
        vectorizer = TfidfVectorizer(max_features=num_features)
        tfidf_matrix = vectorizer.fit_transform(
            non_empty_descriptions["description"].to_list()
        )

        tfidf_df = pl.DataFrame(
            tfidf_matrix.toarray(),
            schema=[f"tfidf_{word}" for word in vectorizer.get_feature_names_out()],
        )

        cars_combined = pl.concat(
            # once we've extracted data out of decrption, we will drop,
            # as this a very high cardinatility variable.
            [non_empty_descriptions.drop("description"), tfidf_df],
            how="horizontal",
        )
        # diagonals a bit weird here. but basically join vertically but you can dont have to give
        # cols in exact right order.
        return pl.concat([cars_combined, empty_descriptions], how="diagonal_relaxed")

    print("No non-empty descriptions to process.")
    return empty_descriptions
