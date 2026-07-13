import pandas as pd
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from sklearn.model_selection import train_test_split

   def preprocess_data(data_path):
       # Load the data
       data = pd.read_csv(data_path)
       
       # Perform data cleaning and preprocessing
       data = data.dropna()  # Remove missing values
       
       # Encode categorical variables
       label_encoder = LabelEncoder()
       data['category'] = label_encoder.fit_transform(data['category'])
       
       # Scale numerical features
       scaler = StandardScaler()
       data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])
       
       # Split the data into features and target
       X = data.drop(['target'], axis=1)
       y = data['target']
       
       # Split the data into train and test sets
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
       return X_train, X_test, y_train, y_test

   # Usage example
   data_path = 'path/to/your/data.csv'
   X_train, X_test, y_train, y_test = preprocess_data(data_path)
