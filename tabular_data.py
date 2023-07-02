import pandas as pd
import ast
import re

def remove_rows_with_missing_ratings(listings):
    listings = listings.drop('Unnamed: 19', axis=1)
    listings = listings.dropna(subset=['Value_rating'])
    
    return listings

def combine_description_strings(listings):
    
    listings = listings.dropna(subset=['Description'])
    
    # deleting 1 listing - poor data entry
    listings = listings[listings['ID'] != '4c917b3c-d693-4ee4-a321-f5babc728dc9']
    
    # strings of a list of strings changed to list of strings
    listings['Description'] = listings['Description'].apply(ast.literal_eval)
    
    delete_these_parts = ['About this space', 'The space', 'Other things to note', 'Guest access', '']
    for part in delete_these_parts:
        listings['Description'] = listings['Description'].apply(lambda x: [item for item in x if item != part])
    
    listings['Description'] = listings['Description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    
    return listings

def combine_amenities_strings(listings):
    
    # strings of a list of strings changed to list of strings
    listings['Amenities'] = listings['Amenities'].apply(ast.literal_eval)
    
    #all 'Unavailable: TV/nTV' elements to 'Unavailable: TV', delete newline and second repeat
    listings['Amenities'] = listings['Amenities'].apply(lambda amenities: [item.split('\n')[0] if item.startswith('Unavailable') else item for item in amenities])

    delete_these_parts = ['What this place offers', '']
    for part in delete_these_parts:
        listings['Amenities'] = listings['Amenities'].apply(lambda x: [item for item in x if item != part])
        
    listings['Amenities'] = listings['Amenities'].apply(lambda x: '. '.join(x) if isinstance(x, list) else x)
    
    return listings


def set_default_feature_values(listings):
    default_value = 1
    features = ["guests", "beds", "bathrooms", "bedrooms"]
    
    for feature in features:
        listings[feature] = listings[feature].fillna(default_value)
    
    return listings

def replace_newlines(text):
    # Replace consecutive newlines with a dot and a space if no non-alphanumeric character in front
    text = re.sub(r'(?<!\W)\n+', '. ', text)
    
    # Replace consecutive newlines with a single space if a non-alphanumeric character is in front
    text = re.sub(r'(?<=\W)\n+', ' ', text)
    
    return text

def clean_tabular_data(listings):
    listings = remove_rows_with_missing_ratings(listings)
    listings = combine_description_strings(listings)
    listings = set_default_feature_values(listings)
    listings['Description'] = listings['Description'].apply(replace_newlines)
    listings = combine_amenities_strings(listings)
    
    return listings

def load_airbnb(label):
    clean_data = pd.read_csv('tabular_data/clean_tabular_data.csv')

    num_columns = clean_data.select_dtypes(exclude=['object']).columns

    num_features = clean_data[num_columns].drop(columns=[label])

    labels = clean_data[label]

    return num_features, labels


if __name__ == "__main__":
    listings = pd.read_csv('tabular_data/listing.csv')
    cleaned_tab_data = clean_tabular_data(listings)
    cleaned_tab_data.to_csv("tabular_data/clean_tabular_data.csv", index=False)
