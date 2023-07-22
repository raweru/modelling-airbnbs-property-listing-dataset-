import pandas as pd
import ast
import re
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder


def remove_rows_with_missing_ratings(listings):
    
    '''The function removes rows with missing ratings from a given dataframe.
    
    Parameters
    ----------
    listings
        The parameter "listings" is a DataFrame object that contains data about listings. It likely has
    multiple columns, including one named "Value_rating" and possibly a column named "Unnamed: 19". The
    function is designed to remove rows from the DataFrame where the "Value_rating" column has missing
    values
    
    Returns
    -------
        the updated "listings" dataframe after removing rows with missing ratings.
    
    '''
    
    listings = listings.drop('Unnamed: 19', axis=1)
    listings = listings.dropna(subset=['Value_rating'])
    
    
    return listings


def combine_description_strings(listings):
    
    '''The function `combine_description_strings` takes a DataFrame of listings, drops rows with missing
    descriptions, removes a specific listing, converts strings of lists to actual lists, removes
    specific parts from the description, and joins the remaining description strings.
    
    Parameters
    ----------
    listings
        The parameter "listings" is a DataFrame containing information about different listings. It is
    assumed that the DataFrame has a column named "Description" which contains strings describing each
    listing.
    
    Returns
    -------
        the modified "listings" dataframe.
    
    '''
    
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
    
    '''The function `combine_amenities_strings` takes a DataFrame `listings` and performs several
    operations to clean and combine the strings in the 'Amenities' column.
    
    Parameters
    ----------
    listings
        The parameter "listings" is a DataFrame that contains information about different listings. It
    likely has columns such as "Amenities" which contains a list of amenities for each listing. The
    function "combine_amenities_strings" takes this DataFrame as input and performs some operations on
    the "Amen
    
    Returns
    -------
        the modified 'listings' dataframe with the 'Amenities' column updated.
    
    '''
    
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
    
    '''The function sets default values for specific features in a listings dataset.
    
    Parameters
    ----------
    listings
        The parameter "listings" is a dictionary or a data structure that contains information about
    different listings. Each listing has features such as "guests", "beds", "bathrooms", and "bedrooms".
    
    Returns
    -------
        the updated "listings" dictionary with default values filled in for any missing values in the
    "guests", "beds", "bathrooms", and "bedrooms" features.
    
    '''
    
    default_value = 1
    features = ["guests", "beds", "bathrooms", "bedrooms"]
    
    for feature in features:
        listings[feature] = listings[feature].fillna(default_value)
    
    
    return listings


def replace_newlines(text):
    
    '''The function `replace_newlines` replaces consecutive newlines in a text with either a dot and a
    space or a single space, depending on whether there is a non-alphanumeric character in front of the
    newlines.
    
    Parameters
    ----------
    text
        The `text` parameter is a string that represents the text that you want to modify.
    
    Returns
    -------
        the modified text with consecutive newlines replaced either with a dot and a space or with a single
    space, depending on whether there is a non-alphanumeric character in front of the newlines.
    
    '''
    
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
    
    # standardize the continuous numeric features
    numeric_columns = listings[['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count']]
    scaler = StandardScaler()
    scaler.fit(numeric_columns)
    standardized_columns = scaler.transform(numeric_columns)
    standardized_data = listings.copy()
    standardized_data[numeric_columns.columns] = standardized_columns
    
    # Encode ordinal features
    standardized_data['bathrooms'] = standardized_data['bathrooms'].astype(int)
    standardized_data['guests'] = standardized_data['guests'].astype(int)
    standardized_data['beds'] = standardized_data['beds'].astype(int)
    standardized_data['bedrooms'] = standardized_data['bedrooms'].astype(int)

    columns_to_encode = ['guests', 'beds', 'bathrooms', 'bedrooms']
    encoder = OrdinalEncoder()
    encoded_values = encoder.fit_transform(standardized_data[columns_to_encode])
    standardized_data[columns_to_encode] = encoded_values
    standardized_data[columns_to_encode] = standardized_data[columns_to_encode].astype(int)

    # Encode 'Category' labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(standardized_data['Category'])
    standardized_data['Category'] = encoded_labels
    
    return standardized_data


def load_airbnb(label, num_only=True):
    
    clean_data = pd.read_csv('tabular_data/clean_tabular_data.csv')
    clean_data.reset_index(drop=True, inplace=True)  # Reset the index

    labels = clean_data[label].values  # Convert to NumPy array

    if num_only:
        num_columns = clean_data.select_dtypes(exclude=['object']).columns

        if label in num_columns:
            features = clean_data[num_columns].drop(columns=[label]).values  # Convert to NumPy array
        else:
            features = clean_data[num_columns].values  # Convert to NumPy array

    else:
        features = clean_data.drop(columns=[label]).values  # Convert to NumPy array

    return features, labels



if __name__ == "__main__":
    listings = pd.read_csv('tabular_data/listing.csv')
    cleaned_tab_data = clean_tabular_data(listings)
    cleaned_tab_data.to_csv("tabular_data/clean_tabular_data.csv", index=False)
