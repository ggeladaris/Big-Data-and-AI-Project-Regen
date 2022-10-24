import pickle
import numpy as np
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import math

app = FastAPI()

## We Loa the pickled RF prediction model and our Min Max Scaler
@app.on_event('startup')
def load_model():
    global scaler, regressor 
    pickle_in = open('RFRegressor.pkl','rb')
    pickle_scale = open('MMScaler.pkl', 'rb')
    scaler = pickle.load(pickle_scale)
    regressor = pickle.load(pickle_in)


@app.get('/')
def main():
    return {'Welcome to our Prediction Model API made for ReGeneration Academy on Big Data & Artificial Intelligence (powered by Microsoft). For predictions use /predict'}
            

class new_data(BaseModel):
    host_is_superhost : str
    host_listings_count : float
    host_verifications : str
    host_has_profile_pic : str
    host_identity_verified : str
    room_type : str
    accommodates : float
    bathrooms_text : str
    bedrooms : float
    beds : float
    amenities : str
    minimum_nights : float
    maximum_nights : float
    maximum_nights_avg_ntm : float
    has_availability : str
    availability_30 : float
    availability_365 : float
    number_of_reviews : float
    number_of_reviews_l30d : float
    instant_bookable : str
    calculated_host_listings_count_entire_homes : float
    


def list_encoder(list_as_str):
    li = list(list_as_str.split(" "))
    return len(li)

def bath_encoder(bathrooms_textt):
    dict_bath={'0 baths': 0, '0 shared baths': 0, '1 bath': 6, '1 private bath': 5, '1 shared bath': 4, '1.5 baths': 8, '1.5 shared baths': 7, '12 baths': 27, '2 baths': 10, '2 shared baths': 9, '2.5 baths': 12, '2.5 shared baths': 11, '3 baths': 14, '3 shared baths': 13, '3.5 baths': 16, '3.5 shared baths': 15, '4 baths': 18, '4 shared baths': 17, '4.5 baths': 19, '5 baths': 21, '5 shared baths': 20, '5.5 baths': 22, '6 baths': 23, '6.5 baths': 24, '7 baths': 25, '9.5 baths': 26, 'Half-bath': 3, 'Shared half-bath': 2}
    return dict_bath[bathrooms_textt]

def clean_tf(entry):
    if entry == 't':
        entry = 1
    else:
        entry = 0
    return entry

def clean_room_type(entry):
    if entry == 'Shared room':
        entry = 0
    elif entry == 'Private room':
        entry = 1
    elif entry == 'Entire home/apt':
        entry = 2 
    else:
        entry = 3
    return entry 



@app.post('/predict')
def predict_price(data : new_data):
    data = data.dict()
    data['host_is_superhost'] = clean_tf(data['host_is_superhost'])
    data['host_verifications'] = list_encoder(data['host_verifications'])
    data['host_has_profile_pic'] = clean_tf(data['host_has_profile_pic'])
    data['host_identity_verified'] = clean_tf(data['host_identity_verified'])
    data['room_type'] = clean_room_type(data['room_type'])
    data['bathrooms_text'] = bath_encoder(data['bathrooms_text'])
    data['instant_bookable'] = clean_tf(data['instant_bookable'])
    data['amenities'] = list_encoder(data['amenities'])
    data['has_availability'] = clean_tf(data['has_availability'])
    data['instant_bookable'] = clean_tf(data['instant_bookable'])
    
    a = list(data.values())
    a = scaler.transform([a])
    prediction = regressor.predict(a)
    a = f"Your rental unit's price is: {math.exp(prediction[0])}"
    return a
    
# if __name__ == '__main__':
#     print('yo')
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
    
