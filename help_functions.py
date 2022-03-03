from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

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
                         'Work_of_casuality',
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



def preprocessing(df):
    df = df.replace('Unknown', np.nan)
    df = df.replace('na', np.nan)
    df = df.drop(['Work_of_casuality'], axis=1)    
    df.Time = pd.to_datetime(df.Time).dt.hour
    
    for col in cat_col_with_order:
        df[col] = df[col].map(map_dicts[col])
    
    cat_cols = df.select_dtypes(include='object').columns    
    num_cols = df.select_dtypes(exclude='object').columns    
    num_vals = df.select_dtypes(exclude='object').to_numpy()
    
    label_lst = []
    for col in cat_cols:
        le = LabelEncoder()
        nan_label = df[col].nunique()        
        labels = le.fit_transform(df[col])
        labels = np.where(labels==nan_label, np.nan, labels)
        label_lst.append(labels)
    cat_labels = np.array(label_lst).transpose()
    label_encoded = np.concatenate((cat_labels, num_vals), axis=1)
    
    knn_imputer = KNNImputer(n_neighbors=5, 
                             weights="uniform", 
                             metric='nan_euclidean')
    imputed_data = knn_imputer.fit_transform(label_encoded)
    df_clean = pd.DataFrame(imputed_data)
    
    df_clean.columns = np.concatenate((cat_cols, num_cols))  
    
    return df_clean