from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

   def train_predictive_model(X_train, y_train):
       # Create and train the predictive model
       model = RandomForestClassifier(n_estimators=100, random_state=42)
       model.fit(X_train, y_train)
       
       return model

   def evaluate_model(model, X_test, y_test):
       # Make predictions on the test set
       y_pred = model.predict(X_test)
       
       # Calculate evaluation metrics
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred)
       recall = recall_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred)
       
       return accuracy, precision, recall, f1

   # Usage example
   model = train_predictive_model(X_train, y_train)
   accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
   print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
