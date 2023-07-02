import pandas as pd
import ast


def remove_rows_with_missing_ratings(listings):
    listings = listings.drop('Unnamed: 19', axis=1)
    listings = listings.dropna(subset=['Value_rating'])
    
    return listings

def combine_description_strings(listings):
    
    listings = listings.dropna(subset=['Description'])
    
    # 1 Description cell is a string, not a list of strings
    listings.loc[listings['ID'].str.contains('4c917b3c-d693-4ee4-a321-f5babc728dc9'), 'Description'] = "['Sleeps 6 with pool.']"
    
    listings['Description'] = listings['Description'].apply(ast.literal_eval)
    
    delete_these_parts = ['About this space', 'The space', 'Other things to note', 'Guest access', '']
    for part in delete_these_parts:
        listings['Description'] = listings['Description'].apply(lambda x: [item for item in x if item != part])
    
    listings['Description'] = listings['Description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    
    return listings

def set_default_feature_values(listings):
    default_value = 1
    features = ["guests", "beds", "bathrooms", "bedrooms"]
    
    for feature in features:
        listings[feature] = listings[feature].fillna(default_value)
    
    return listings

def clean_tabular_data(listings):
    listings = remove_rows_with_missing_ratings(listings)
    listings = combine_description_strings(listings)
    listings = set_default_feature_values(listings)
    
    return listings

def load_airbnb(label):
    listings = pd.read_csv('tabular_data/listing.csv')

    non_text_columns = listings.select_dtypes(exclude=['object']).columns

    features = df[non_text_columns].drop(columns=[label])

    labels = df[label]

    return features, labels


if __name__ == "__main__":
    listings = pd.read_csv('tabular_data/listing.csv')
    cleaned_tab_data = clean_tabular_data(listings)
    cleaned_tab_data.to_csv("tabular_data/clean_tabular_data.csv", index=False)
