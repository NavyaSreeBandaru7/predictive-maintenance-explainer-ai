from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer

   def finetune_language_model(model_name, train_dataset, eval_dataset):
       # Load the pre-trained model and tokenizer
       model = AutoModelForQuestionAnswering.from_pretrained(model_name)
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       
       # Define training arguments
       training_args = TrainingArguments(
           output_dir='./results',
           evaluation_strategy='epoch',
           learning_rate=2e-5,
           per_device_train_batch_size=16,
           per_device_eval_batch_size=16,
           num_train_epochs=3,
           weight_decay=0.01,
       )
       
       # Create the Trainer instance
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=train_dataset,
           eval_dataset=eval_dataset,
           tokenizer=tokenizer,
       )
       
       # Fine-tune the model
       trainer.train()
       
       return model, tokenizer

   # Usage example
   model_name = 'bert-base-uncased'
   train_dataset = ...  # Load and preprocess your training dataset
   eval_dataset = ...  # Load and preprocess your evaluation dataset
   model, tokenizer = finetune_language_model(model_name, train_dataset, eval_dataset)
