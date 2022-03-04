import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

cat_col_with_order = ['Day_of_week', 'Age_band_of_driver', 'Driving_experience', 
                  'Service_year_of_vehicle', 'Age_band_of_casualty', 
                  'Casualty_severity', 'Defect_of_vehicle']


cat_col_without_order = ['Type_of_collision',
                         'Pedestrian_movement',
                         'Weather_conditions',
                         'Casualty_class',
                         'Sex_of_casualty',
                         'Road_surface_conditions',
                         'Sex_of_driver',
                         'Vehicle_movement',                         
                         'Types_of_Junction',
                         'Type_of_vehicle',
                         'Vehicle_driver_relation',
                         'Light_conditions',
                         'Educational_level',
                         'Road_allignment',
                         'Cause_of_accident',
                         'Fitness_of_casuality',
                         'Road_surface_type',
                         'Owner_of_vehicle',
                         'Lanes_or_Medians',
                         'Area_accident_occured']


map_dicts = {
    'Day_of_week': {'Monday': 1, "Tuesday": 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 0},
    'Age_band_of_driver': {'Under 18': 0, '18-30': 1, '31-50': 2, 'Over 51': 3},
    'Driving_experience': {'No Licence': 0, 'Below 1yr': 1, '1-2yr': 2, '2-5yr': 3, '5-10yr': 4, 'Above 10yr': 5},
    'Service_year_of_vehicle': {'Below 1yr': 0, '1-2yr': 1, '2-5yr': 2, '5-10yr': 3, 'Above 10yr': 4},
    'Age_band_of_casualty': {'5': 0, 'Under 18': 1, '18-30': 2, '31-50': 3, 'Over 51': 4},
    'Casualty_severity': {'1': 1, '2': 2, '3': 3},
    'Defect_of_vehicle': {'No defect': 0, '5': 5, '7': 7}
}

def preprocess(df):
    df = df.replace('Unknown', np.nan)
    df = df.replace('na', np.nan)
    df = df.drop(['Work_of_casuality'], axis=1)
    df.Time = pd.to_datetime(df.Time).dt.hour
    
    for col in cat_col_with_order:
        df[col] = df[col].map(map_dicts[col])
    return df

def encode(df, encoder):
    '''function to encode non-null data and replace it in the original data'''      
        
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include='object').columns    
    
    for col in cat_cols:
        #retains only non-null values
        nonulls = np.array(df_enc[col].dropna())
        #reshapes the data for encoding
        impute_reshape = nonulls.reshape(-1,1)
        #encode date
        impute_ordinal = encoder.fit_transform(impute_reshape)
        #Assign back encoded values to non-null values
        df_enc.loc[df_enc[col].notnull(), col] = np.squeeze(impute_ordinal)
    return df_enc

def impute(df, imputer):
    return pd.DataFrame(np.round(imputer.fit_transform(df)), columns = df.columns)


def label_encoder(df):    
    mask = df.isnull()
    le = LabelEncoder    
    labels = df.apply(le.fit_transform)
    df_label_encoded = labels.where(~mask, df)
    
    return df_label_encoded
    
    
