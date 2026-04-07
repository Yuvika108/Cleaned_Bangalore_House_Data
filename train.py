import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Path to the data
data_path = 'cleaned_bangalore_house_data.csv'
data = pd.read_csv(data_path)

# Prepare X and y
X = data.drop(columns=['price'])
y = data['price']

# Create the pipeline
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['location']),
    remainder='passthrough'
)

pipe = make_pipeline(column_trans, StandardScaler(with_mean=False), Ridge())

# Fit the model
print("Fitting the model...")
pipe.fit(X, y)

# Save the model
pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))
print("Model saved to RidgeModel.pkl")
