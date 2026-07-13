from lime import lime_tabular

   def train_xai_model(model, X_train, feature_names):
       # Create the XAI model
       explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['0', '1'], discretize_continuous=True)
       
       return explainer

   def explain_instance(explainer, instance):
       # Generate explanations for a single instance
       explanation = explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_names))
       
       return explanation

   # Usage example
   feature_names = ['feature1', 'feature2', 'category']
   explainer = train_xai_model(model, X_train, feature_names)
   instance = X_test.iloc[0]  # Select a single instance to explain
   explanation = explain_instance(explainer, instance)
   print(explanation.as_list())
