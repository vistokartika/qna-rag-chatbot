import pandas as pd
from tqdm import tqdm
import emoji

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the columns that are not needed
    columns_to_drop = ['Unnamed: 0', 'review_id', 'pseudo_author_id', 'author_name']
    df = df.drop(columns=columns_to_drop)

    # Rename the columns
    df = df.rename(columns={'review_rating': 'rating_from_reviewer', 'author_app_version': 'app_version', 'review_timestamp': 'review_date'})

    # Strip review_date to only keep the date
    df['review_date'] = df['review_date'].str.split(' ').str[0]

    return df

def parse(df: pd.DataFrame) -> pd.DataFrame:
    # Format each column as "column_name: value"
    formatted_columns = [df[col].map(f"{col}: {{}}".format) for col in df.columns]

    # Combine all columns for each row with "; " separator
    optimized_formatted_data = pd.DataFrame(formatted_columns).T.agg('; '.join, axis=1)

    # Display the first few formatted strings as a sample
    row_list = optimized_formatted_data.tolist()

    txt_string = "SPOTIFY REVIEWS FROM GOOGLE STORES\n\n"
    for row in tqdm(row_list):
        # replace emojis with their description
        row = emoji.demojize(row)

        # Remove any non-ASCII characters
        row = row.encode('ascii', 'ignore').decode('ascii')
        
        txt_string += row + '\n'

    return txt_string

if __name__ == '__main__':
    df = pd.read_csv('dataset/SPOTIFY_REVIEWS.csv')
    df = df.tail(10000)
    df = preprocess(df)
    df.to_csv('dataset/SPOTIFY_REVIEWS_CLEANED.csv', index=False)
    result = parse(df)
    with open('dataset/SPOTIFY_REVIEWS_CLEANED.txt', 'w') as file:
        file.write(result)
        