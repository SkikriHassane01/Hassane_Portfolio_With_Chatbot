# TODO: import libraries
import json
import pickle
import os
import sys
import numpy as np
from pathlib import Path

from typing import Dict, List

import nltk
from nltk.stem import WordNetLemmatizer

import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer

import mlflow
import mlflow.keras

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)

# Now we can import from backend modules
from src.utils.logger import setup_logger
from config import Config

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.metrics import Precision, Recall, AUC

# TODO: Define the Chatbot training class
class ChatbotModelTrainer:
    """
    A class to handle the training of an intent classification model using Sentence Transformer embeddings.
    """
    # TODO: initialize the parameters
    def __init__(self,
                 num_epochs: int = 200,
                 use_data_augmentation: bool = True):
        self.num_epochs = num_epochs
        self.use_data_augmentation = use_data_augmentation
        self.logger = setup_logger("train_model_improved")
        self.classes = [] # unique intent categories(tags) ex: ['greeting', 'goodbye', 'thanks']
        self.documents = [] # list of tuples (tokenize sentence, tag) ex: [(['Hi'], 'greeting'), (['How', 'are', 'you'], 'greeting')]
        self.all_patterns = []  # list of all patterns (original intent) ex: ['Hi', 'How are you']
        self.X_data = None # Sentence Transformer embeddings
        self.y_data = None # one-hot encoded intent classes
        self.model = None 
        self.embedding_dim = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Use Config for base directory and paths
        self.base_dir = Config.BASE_DIR
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.base_dir / 'models'
        
        # Create required directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger.info(f">>> Project base directory: {self.base_dir}")
        self.logger.info(f">>> Data directory: {self.data_dir}")
        self.logger.info(f">>> Models directory: {self.models_dir}")
        
        # Download NLTK resources
        self._download_nltk_resources()
    
    # TODO: Define the helper download nltk resources function   
    def _download_nltk_resources(self):
        """
        Download the required NLTK resources.
        """
        self.logger.info(">> Downloading NLTK resources")
        resources = ['punkt', 'wornet','omw-1.4', 'averaged_perceptron_tagger', 'tagsets']
        try:
            for resource in resources:
                nltk.download(resources, quiet=True)
            self.logger.info('>> The NLTK resources was downloaded successfully')
        except Exception as e:
            self.logger.error(f'ERROR while downloading the NLTK resources: {str(e)}')
     
    # TODO: Load the data from the intents.json file    
    def load_data(self, data_path:str='intents.json') -> Dict:
        """
        Load intents data from json file
        Args:
            - data_path: path to the intent json file
        Return:
            - Dictionary containing the loaded intents
        """
        # Use Config path for intents file
        full_path = self.base_dir / Config.INTENTS_PATH
        self.logger.info(f">>> Loading intents file from {full_path}")
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                intents = json.load(file)
            self.logger.info(f"Intents file loaded successfully with {len(intents['intents'])} intent categories")
            return intents
        except Exception as e:
            self.logger.error(f'Error while loading the intents json file: {str(e)}')
    
    # data processing >> tokenization, lemmatization,     
    def process_data(self, intents: Dict) -> None:
        """
        Process intents and patterns to create training data
        
        Args:
            - intents: Dict containing the intents data
        """
        self.logger.info("Processing Data Started")
        
        for intent in intents["intents"]:
            for pattern in intent['patterns']:
                if isinstance(pattern, list):
                    pattern = ' '.join(pattern)
                elif not isinstance(pattern, str):
                    pattern = str(pattern)
                    
                #TODO: Tokenize each pattern
                word_tokens = nltk.word_tokenize(pattern)
                lemmatized_tokens = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokens if not ['.', '?', '!', ',']]
                self.documents.append((lemmatized_tokens, intent['tag']))
                self.all_patterns.append(pattern)
                
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        #TODO: remove the duplicated and sort the classes
        self.classes = sorted(list(set(self.classes)))
        
        self.logger.info(f">> Total patterns: {len(self.all_patterns)}")
        self.logger.info(f">> Total Intent Classes: {len(self.classes)}")
        
        mlflow.log_metric("Original_pattern_count", len(self.all_patterns))
        mlflow.log_metric("Intent_class_count", len(self.classes))

    # TODO: Augment the data with synonymous data
    def augment_data(self, augmentation_factor:5)-> None:
        """
        Perform data augmentation to increase the training examples 
        
        Args:
            - augmentation_factor: Number of augmented examples to create per original
        """
        
        if not self.use_data_augmentation:
            return

        self.logger.info(">> Performing Data Augmentation ....")
        
        try:
            aug = naw.SynonymAug()
            
            augmented_documents = []
            augmented_patterns = []
            
            for i, (doc, label) in enumerate(self.documents):
                #keep the original document
                augmented_documents.append((doc, label))
                augmented_patterns.append(self.all_patterns[i])
                
                # Create the augmented version
                text = self.all_patterns[i]
                for _ in range(augmentation_factor):
                    try:   
                        augmented_text = aug.augment(text)
                        # ensure that the augmented text is a string
                        if isinstance(augmented_text, list):
                            augmented_text = ' '.join(augmented_text) #  ["how", "are", "you", "doing"] → "how are you doing"
                        augmented_tokens = nltk.word_tokenize(augmented_text)
                        augmented_documents.append((augmented_tokens, label))
                        augmented_patterns.append(augmented_text)
                    except Exception as e:
                        self.logger.warning(f"Augmentation failed for '{text}': {str(e)}")
            self.logger.info(f"After augmentation: {len(augmented_documents)} patterns (was {len(self.documents)})")
            mlflow.log_metric("augmented_pattern_count", len(augmented_documents))
            self.documents = augmented_documents
            self.all_patterns = augmented_patterns
        except Exception as e:
            self.logger.error(f"Data augmentation failed, using original data: {str(e)}")
     
    # TODO: Create the embeddings and prepare the X_data and y_data data (embedding and one-hot-encoding)           
    def create_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Create embeddings for patterns using SentenceTransformer.
        
        Args:
            - model_name: Name of the SentenceTransformer Model to use
        """
        self.logger.info(f">> Loading Sentence Transformer Model {model_name} ....")
        model_st = SentenceTransformer(model_name)
        
        self.logger.info(">> Create embeddings for patterns ...")
        
        # create embeddings for all the patterns
        self.X_data = model_st.encode(self.all_patterns)
        self.embedding_dim = self.X_data.shape[1]

        # create one-hot-encoding outputs for intent classes
        y_data = []
        for doc, intent_tag in self.documents:
            output_row = [0] * len(self.classes)
            output_row[self.classes.index(intent_tag)] = 1
            y_data.append(output_row)
        
        self.y_data = np.array(y_data)
        
        # Log embedding dimension
        mlflow.log_param("embedding_dim", self.embedding_dim)
        
        # save the model information
        model_info = {
            "embedding_method": 'sentence_transformer',
            "model_name" : model_name,
            "embedding_dim": self.embedding_dim,
            "num_classes": len(self.classes)
        }
        
        try:
            # Use the path from Config
            model_info_path = self.base_dir / Config.MODEL_INFO
            model_info_path.parent.mkdir(exist_ok=True)
            
            with open(model_info_path, 'wb') as file:
                pickle.dump(model_info, file)
                self.logger.info(f">> Model info saved to {model_info_path}")
        except Exception as e:
            self.logger.error(f"Error while saving model info: {str(e)}")
         
        # shuffle data
        """
        in this step we want to prevents the model from learning patterns based on the order of examples and helps ensure each
        training contains a mix of different classes.
        """
        indices = np.random.permutation(len(self.X_data))
        self.X_data = self.X_data[indices]
        self.y_data = self.y_data[indices]
        
    #TODO: build the architecture of our model
    def build_model(self) -> None:
        """Build the neural network model architecture."""
        self.logger.info("Building Sequential model with Sentence Transformer embeddings...")
        
        # Model with Sentence Transformer embeddings
        self.model = Sequential(
            [
                # input layer
                Dense(256, input_shape=(self.embedding_dim,), activation='relu', kernel_regularizer= l2(0.001)),
                BatchNormalization(),
                Dropout(0.5),
                
                # hidden layer
                Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
                BatchNormalization(),
                Dropout(0.4),
                
                # output layer
                Dense(len(self.classes), activation='softmax')
            ]
        )
        
        # compile the model
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer = optimizer,
            metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
            ]
        )
        """
        Metrics Explanation:
            - Accuracy ==> Hom many predictions are correct?
            - AUC ==> How well can the model distinguish between intents?
            - Precision ==> When the model predicts an intent, how often is it right?  >> Avoid wrong predictions(false positive) are costly
            - Recall ==> Of all actual instances of intent, how many did we catch?  >> When missing an intent is costly
        """
        
        self.logger.info(f'Model build successfully with architecture: {self.model.summary()}')
        

    def define_callbacks(self) -> List:
        """
        Define callbacks for model training.
        
        Returns:
            List of callback objects for training
        """
        
        class CustomCallback(keras.callbacks.Callback):
            """
            Print detailed metrics at the end of each epoch
            """
            def on_epoch_end(self, epoch, logs=None):
                """
                Print metrics at the end of each epoch
                """
                print(f"\nEpoch: {epoch + 1}")
                print("-" * 60)
                print(f"Training Metrics:")
                print(f"|---Loss: {logs['loss']:.6f}")
                print(f"|---Accuracy: {logs['accuracy']:.6f}")
                print(f"|---AUC: {logs['auc']:.6f}")
                print(f"|---Precision: {logs['precision']:.6f}")
                print(f"|---Recall: {logs['recall']:.6f}")
                
                print("\n")
                print("-" * 60)
                print(f"Validation Metrics:")
                print(f"|---Val Loss: {logs['val_loss']:.6f}")
                print(f"|---Val Accuracy: {logs['val_accuracy']:.6f}")
                print(f"|---Val AUC: {logs['val_auc']:.6f}")
                print(f"|---Val Precision: {logs['val_precision']:.6f}")
                print(f"|---Val Recall: {logs['val_recall']:.6f}")
                print("-" * 60)
                        
        callbacks = [
            CustomCallback(),
            
            EarlyStopping(
                monitor='val_accuracy',
                patience=30,
                min_delta=0.0001,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', # loss is more sensitive to small changes than accuracy
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
        ]
        
        return callbacks
    
    # TODO: train the model
    def train_model(self, validation_split: float = 0.2) -> Dict:
        """
        Train the model with the prepared data.
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training model (this may take a while)...")
        mlflow.log_param("validation_split", validation_split)
        
        callbacks = self.define_callbacks()
        
        hist = self.model.fit(
            self.X_data, self.y_data,
            epochs = self.num_epochs,
            batch_size=16,
            verbose=0,  # Hide the progress bar
            validation_split=validation_split,
            callbacks= callbacks
        )
        
        # Get best metrics from the best validation accuracy epoch
        if 'val_accuracy' in hist.history:
            best_val_accuracy_idx = np.argmax(hist.history['val_accuracy'])
            best_val_accuracy = hist.history['val_accuracy'][best_val_accuracy_idx]
            best_val_loss = hist.history['val_loss'][best_val_accuracy_idx]
            best_val_auc = hist.history['val_auc'][best_val_accuracy_idx]
            best_val_precision = hist.history['val_precision'][best_val_accuracy_idx]
            best_val_recall = hist.history['val_recall'][best_val_accuracy_idx]
            
            # Get training metrics from the same epoch as best validation
            best_epoch_train_acc = hist.history['accuracy'][best_val_accuracy_idx]
            best_epoch_train_loss = hist.history['loss'][best_val_accuracy_idx]
            best_epoch_train_auc = hist.history['auc'][best_val_accuracy_idx]
            best_epoch_train_precision = hist.history['precision'][best_val_accuracy_idx]
            best_epoch_train_recall = hist.history['recall'][best_val_accuracy_idx]
            
            self.logger.info(f"Best Validation accuracy: {best_val_accuracy} at epoch {best_val_accuracy_idx + 1}")
        else:
            best_val_accuracy = None
            best_val_loss = None
            best_val_auc = None
            best_val_precision = None
            best_val_recall = None
            best_val_accuracy_idx = -1
            
            best_epoch_train_acc = max(hist.history['accuracy'])
            best_epoch_train_loss = min(hist.history['loss'])
            best_epoch_train_auc = max(hist.history['auc'])
            best_epoch_train_precision = max(hist.history['precision'])
            best_epoch_train_recall = max(hist.history['recall'])
        

        return {
            'train_accuracy': best_epoch_train_acc,
            'train_loss': best_epoch_train_loss,
            'train_auc': best_epoch_train_auc,
            'train_precision': best_epoch_train_precision,
            'train_recall': best_epoch_train_recall,
            'val_accuracy': best_val_accuracy,
            'val_loss': best_val_loss,
            'val_auc': best_val_auc,
            'val_precision': best_val_precision,
            'val_recall': best_val_recall,
            'best_epoch': best_val_accuracy_idx + 1,
            'epochs_trained': len(hist.epoch)
        }

    # TODO: Save the model
    def save_model(self, model_name: str = "chatbot_model.h5") -> None:
        """
        Save the model and all necessary artifacts for later use
        """
        try:
            # Save model architecture and weights using Config path
            model_path = self.base_dir / Config.MODEL_PATH
            model_path.parent.mkdir(exist_ok=True)
            self.model.save(model_path)
            self.logger.info(f">> Model saved to {model_path}")
            
            # Save classes list for prediction using Config path
            classes_path = self.base_dir / Config.CLASSED_PATH
            classes_path.parent.mkdir(exist_ok=True)
            with open(classes_path, 'wb') as f:
                pickle.dump(self.classes, f)
            self.logger.info(f">> Class labels saved to {classes_path}")
            
            # Save model info including embedding details using Config path
            model_info = {
                "embedding_method": 'sentence_transformer',
                "embedding_dim": self.embedding_dim,
                "num_classes": len(self.classes),
                'num_patterns': len(self.documents),
                "classes": self.classes
            }
            
            model_info_path = self.base_dir / Config.MODEL_INFO
            model_info_path.parent.mkdir(exist_ok=True)
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f)
            self.logger.info(f">> Model info saved to {model_info_path}")
                
            # Log model with MLflow including all necessary files
            mlflow.log_artifact(str(model_path), "model")
            mlflow.log_artifact(str(classes_path), "model")
            mlflow.log_artifact(str(model_info_path), "model")
            
            self.logger.info("Model and artifacts successfully logged to MLflow")
            
        except Exception as e:
            self.logger.error(f"Error while saving model artifacts: {str(e)}")
            raise

    # TODO: Main method
    def train(self) -> Dict:
        """
        Main method to execute the entire training workflow
        
        Returns:
            - Dict with training results
        """
        self.logger.info(">>>> Starting model training process with sentence transformer embeddings ...")
        
        # setup MLflow tracking - setting local directory for tracking
        mlflow.set_tracking_uri("file:" + str(self.base_dir / "mlruns"))
        
        # setup the workflow
        mlflow.set_experiment("Chatbot intent classification")
        
        # start MLflow run
        with mlflow.start_run(run_name="Sentence_transformer_model"):
            #log parameters 
            mlflow.log_param('embedding_method', "sentence_transformer")
            mlflow.log_param('num_epochs', self.num_epochs)
            mlflow.log_param("use_data_augmentation", self.use_data_augmentation)

            # load and process data
            intents = self.load_data()
            self.process_data(intents)
            
            # augment data if enabled
            if self.use_data_augmentation:
                self.augment_data(augmentation_factor=10)
            
            # create embeddings
            self.create_embeddings()
            
            # Build model
            self.build_model()
            
            # Train Model
            result = self.train_model()
            
            # Log all metrics from the training result
            self.logger.info("Logging final model metrics to MLflow...")
            mlflow.log_metric("final_train_accuracy", result['train_accuracy'])
            mlflow.log_metric("final_train_loss", result['train_loss'])
            mlflow.log_metric("final_train_auc", result['train_auc'])
            mlflow.log_metric("final_train_precision", result['train_precision'])
            mlflow.log_metric("final_train_recall", result['train_recall'])
            
            if result['val_accuracy'] is not None:
                mlflow.log_metric("final_val_accuracy", result['val_accuracy'])
                mlflow.log_metric("final_val_loss", result['val_loss'])
                mlflow.log_metric("final_val_auc", result['val_auc'])
                mlflow.log_metric("final_val_precision", result['val_precision'])
                mlflow.log_metric("final_val_recall", result['val_recall'])
            
            mlflow.log_metric("best_epoch", result['best_epoch'])
            mlflow.log_metric("total_epochs", result['epochs_trained'])
            
            
            # save the model
            self.save_model()
        
            return result

if __name__ == "__main__":
    trainer = ChatbotModelTrainer(num_epochs=300, use_data_augmentation=True)
    result = trainer.train()
    
    print("\n" + "="*70)
    print(f"FINAL MODEL PERFORMANCE METRICS (Best Epoch: {result['best_epoch']})")
    print("="*70)
    print("Training Metrics:")
    print(f"  - Accuracy:  {result['train_accuracy']:.4f}")
    print(f"  - Loss:      {result['train_loss']:.4f}")
    print(f"  - AUC:       {result['train_auc']:.4f}")
    print(f"  - Precision: {result['train_precision']:.4f}")
    print(f"  - Recall:    {result['train_recall']:.4f}")
    
    print("\nValidation Metrics:")
    print(f"  - Accuracy:  {result['val_accuracy']:.4f}")
    print(f"  - Loss:      {result['val_loss']:.4f}")
    print(f"  - AUC:       {result['val_auc']:.4f}")
    print(f"  - Precision: {result['val_precision']:.4f}")
    print(f"  - Recall:    {result['val_recall']:.4f}")
    print("="*70)