import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder




def  unuseful_labels(df: pd.DataFrame)-> pd.DataFrame:    
    """
        Drops the features that are considered unhelpful
        
        :param df: input dataFrame
        :return : a copy of dataFrame without some columns
    """
    df_copy =df.drop(labels=['host_response_time','host_response_rate','host_acceptance_rate',
        'reviews_per_month','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms',
        'id','listing_url','scrape_id','last_scraped','name','description','neighborhood_overview',
        'picture_url','host_id', 'host_url','host_name', 'host_since', 'host_location', 'host_about',
        'host_thumbnail_url', 'host_picture_url','host_neighbourhood','neighbourhood','neighbourhood_group_cleansed',
        'latitude','longitude','bathrooms','calendar_updated','calendar_last_scraped','first_review', 'last_review',
        'license','host_total_listings_count','minimum_minimum_nights', 'maximum_minimum_nights',
        'minimum_maximum_nights', 'maximum_maximum_nights',
        'minimum_nights_avg_ntm', 'availability_60','number_of_reviews_ltm','calculated_host_listings_count',
        'availability_90','property_type'],axis=1)
    return df_copy


def preprocess_the_prices(df : pd.DataFrame) -> pd.DataFrame:
    """
       :param df: input dataFrame
       :return : a copy of dataFrame with clean type of prices
    """
    df_copy = df.copy()
    df_copy['price']=df_copy['price'].str.strip('$')
    df_copy['price'] = df_copy['price'].apply(lambda x: float(x.split()[0].replace(',', '')))
    return df_copy


def encoding(df:pd.DataFrame)->pd.DataFrame:
    """
        Encodes the following: room-types, baths, host related features and amenities
        
        param df: Input DataFrame
        return: The DataFrame after encoding
    """
    df_copy=df.copy()
    df_copy['room_type'] = df_copy['room_type'].map(lambda x: 0 if x=='Shared room' else (1 if x=='Private room' else(2 if x=='Entire home/apt' else 3)))
    np.unique(np.array(df_copy['room_type'].astype('str')),return_counts=True)
   
    
    # for the other features
    
    le=LabelEncoder()
    df_copy['host_is_superhost']=le.fit_transform(df_copy['host_is_superhost'])
    df_copy['host_has_profile_pic']=le.fit_transform(df_copy['host_has_profile_pic'])
    df_copy['host_identity_verified']=le.fit_transform(df_copy['host_identity_verified'])
    df_copy['has_availability']=le.fit_transform(df_copy['has_availability'])
    df_copy['instant_bookable']=le.fit_transform(df_copy['instant_bookable'])
    df_copy['neighbourhood_cleansed']=le.fit_transform(df_copy['neighbourhood_cleansed'])
    #  for room type # 'Entire home/apt'-3, 'Hotel room'-2, 'Private room'-1, 'Shared room'-0
    df_copy['room_type'] = df_copy['room_type'].map(lambda x: 0 if x=='Shared room' else (1 if x=='Private room' else(2 if x=='Entire home/apt' else 3)))
    np.unique(np.array(df['room_type'].astype('str')),return_counts=True)
    
    # for the bathrooms
    cat=np.unique(np.array(df_copy['bathrooms_text'].astype('str')))[0:-1]
    cat[-1]='0.3 '+cat[-1]
    cat[-2]='0.5 '+cat[-2]
    dict={}
    
    for i,elem in enumerate(cat):
        if 'shared' in elem:
            dict[elem]=float(''.join(i for i in elem if (i.isdigit() or i == ".")))-0.2
        elif 'private' in elem:
            dict[elem]=float(''.join(i for i in elem if (i.isdigit() or i == ".")))-0.1
        else:
            dict[elem]=float(''.join(i for i in elem if (i.isdigit() or i == ".")))
    
    a = sorted(dict.items(), key=lambda x: x[1])
    j=0
    for i in a:
        dict[i[0]]=j
        j+=1

    dict['0 baths']=0
    dict['Half-bath'] = dict['0.5 Half-bath']
    dict['Shared half-bath']=dict['0.3 Shared half-bath']
    del dict['0.3 Shared half-bath']
    del dict['0.5 Half-bath']
    dict['nan']=np.NaN

    df_copy['bathrooms_text'].replace(dict,inplace=True)

    # for the host
    df_copy['host_verifications'] = df_copy['host_verifications'].apply(lambda x: ast.literal_eval(x))
    df_copy['host_verifications'] = df_copy['host_verifications'].fillna(value=np.nan)
    for y in range(9582):
        if type(df_copy['host_verifications'][y])==list:
            df_copy.at[y,'host_verifications'] = len(df_copy['host_verifications'][y])

    # for the amenities
    df_copy['amenities'] = df_copy['amenities'].apply(lambda x: ast.literal_eval(x))
    for y in range(9582):
        if type(df_copy['amenities'][y])==list:
           df_copy.at[y,'amenities'] = len(df_copy['amenities'][y])
    return df_copy



def one_hot_encoding(df:pd.DataFrame)->pd.DataFrame:
    """
        :param df: input dataFrame
        :return : a copy of dataFrame with the neighbourhood_cleansed feature encoded with onehot encoding
    """
    df_copy=df.copy()
    df_copy = pd.get_dummies(df_copy, prefix=['neighbourhood_cleansed'], columns=['neighbourhood_cleansed'])
    return df_copy

def Preprocessing(df : pd.DataFrame)-> pd.DataFrame:
    """
        :param df:input dataFrame
        :return: a copy of the dataframe with the data preprocessed
    """
    df_c = unuseful_labels(df)
    df_c = preprocess_the_prices(df_c)
    df_c = encoding(df_c)
    df_c = one_hot_encoding(df_c)
    df_c=df_c.apply(pd.to_numeric)
    return df_c
