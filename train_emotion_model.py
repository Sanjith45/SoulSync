"""
Emotion Analysis Model Training Script using GoEmotions Dataset
This script trains a custom emotion classification model using the full GoEmotions dataset from Hugging Face
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os
from collections import Counter
import re
import requests
import gzip

class EmotionAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
        self.model = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
        self.emotion_labels = []
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def load_goemotions_data(self):
        """Load GoEmotions dataset from Hugging Face"""
        print("Loading GoEmotions dataset from Hugging Face...")
        
        try:
            # Try to load from Hugging Face datasets
            from datasets import load_dataset
            dataset = load_dataset("go_emotions")
            
            # Combine train and validation sets for training
            train_data = dataset['train']
            val_data = dataset['validation']
            
            print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
            
            # Process the data
            df_train = self._process_goemotions_data(train_data)
            df_val = self._process_goemotions_data(val_data)
            
            # Combine for full training
            df = pd.concat([df_train, df_val], ignore_index=True)
            
            print(f"Processed dataset shape: {df.shape}")
            print(f"Emotion distribution:")
            print(df['emotion'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Error loading GoEmotions dataset: {e}")
            print("Falling back to enhanced synthetic data...")
            return self.create_synthetic_emotion_data()
    
    def _process_goemotions_data(self, dataset):
        """Process GoEmotions dataset format"""
        texts = []
        emotions = []
        
        # GoEmotions labels (28 emotions)
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        for example in dataset:
            text = example['text']
            labels = example['labels']
            
            # Clean the text
            cleaned_text = self._clean_text(text)
            if not cleaned_text or len(cleaned_text.strip()) < 3:
                continue
                
            # Handle multiple labels - use the first non-neutral label, or neutral if none
            emotion = 'neutral'
            for label_idx in labels:
                if label_idx < len(emotion_labels):
                    emotion = emotion_labels[label_idx]
                    if emotion != 'neutral':
                        break
            
            texts.append(cleaned_text)
            emotions.append(emotion)
        
        return pd.DataFrame({'text': texts, 'emotion': emotions})
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text
    
    def create_synthetic_emotion_data(self):
        """Create synthetic training data based on GoEmotions patterns"""
        emotion_examples = {
            'joy': [
                "I'm so happy today!", "This is amazing!", "I love this!", 
                "Fantastic news!", "I'm thrilled!", "Wonderful!", "Great job!",
                "I'm excited about this!", "This makes me smile!", "Perfect!"
            ],
            'sadness': [
                "I'm feeling down", "This is so sad", "I'm depressed", 
                "I feel lonely", "I'm crying", "This hurts", "I'm miserable",
                "I feel empty", "I'm heartbroken", "This is devastating"
            ],
            'anger': [
                "I'm furious!", "This makes me mad", "I'm so angry", 
                "I hate this!", "This is ridiculous!", "I'm outraged",
                "I can't stand this", "This is infuriating", "I'm livid"
            ],
            'fear': [
                "I'm scared", "This is terrifying", "I'm afraid", 
                "I'm worried", "This is frightening", "I'm anxious",
                "I'm nervous", "I'm terrified", "This is scary"
            ],
            'love': [
                "I love you", "You mean everything to me", "I care about you",
                "You're special to me", "I adore you", "You're wonderful",
                "I cherish you", "You're amazing", "I'm grateful for you"
            ],
            'surprise': [
                "Wow!", "I can't believe it!", "This is unexpected", 
                "Really?", "No way!", "I'm shocked", "This is surprising",
                "I didn't see that coming", "Amazing!"
            ],
            'disgust': [
                "This is disgusting", "I hate this", "This is gross", 
                "Yuck!", "This is revolting", "I can't stand this",
                "This is awful", "I'm repulsed", "This is sickening"
            ],
            'gratitude': [
                "Thank you so much", "I'm grateful", "I appreciate this",
                "Thanks a lot", "I'm thankful", "This means a lot",
                "I'm blessed", "I'm fortunate", "I'm lucky"
            ],
            'anxiety': [
                "I'm anxious", "I'm worried", "I'm stressed", 
                "I'm nervous", "I'm concerned", "I'm uneasy",
                "I'm troubled", "I'm restless", "I'm tense"
            ],
            'excitement': [
                "I'm so excited!", "This is thrilling!", "I can't wait!",
                "I'm pumped!", "This is awesome!", "I'm stoked!",
                "I'm hyped!", "This is incredible!", "I'm ecstatic!"
            ],
            'neutral': [
                "The weather is nice", "I went to the store", "It's 3 PM",
                "The book is on the table", "I have a meeting tomorrow",
                "The car is blue", "I need to buy milk", "Today is Monday"
            ]
        }
        
        texts = []
        emotions = []
        
        for emotion, examples in emotion_examples.items():
            for text in examples:
                texts.append(text)
                emotions.append(emotion)
        
        return pd.DataFrame({'text': texts, 'emotion': emotions})
    
    def train(self):
        """Train the emotion classification model"""
        print("Training emotion classification model...")
        
        # Load training data
        df = self.load_goemotions_data()
        
        # Prepare features and labels
        X = df['text']
        y = df['emotion']
        
        # Filter out emotions with too few samples (less than 10)
        emotion_counts = y.value_counts()
        valid_emotions = emotion_counts[emotion_counts >= 10].index
        df_filtered = df[df['emotion'].isin(valid_emotions)]
        
        print(f"Filtered dataset shape: {df_filtered.shape}")
        print(f"Valid emotions: {len(valid_emotions)}")
        
        X = df_filtered['text']
        y = df_filtered['emotion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train the model
        self.model.fit(X_train_vec, y_train_encoded)
        
        # Evaluate the model
        y_pred_encoded = self.model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store emotion labels and create mapping
        self.emotion_labels = list(self.label_encoder.classes_)
        self.emotion_mapping = {emotion: emotion.title() for emotion in self.emotion_labels}
        self.is_trained = True
        
        return accuracy
    
    def predict_emotion(self, text):
        """Predict emotion for a given text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return {
                'emotion': 'Neutral',
                'confidence': 0.0,
                'raw_emotion': 'neutral',
                'method': 'empty_text'
            }
        
        # Vectorize the input text
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Get prediction and probability
        emotion_encoded = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        # Decode emotion
        emotion = self.label_encoder.inverse_transform([emotion_encoded])[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Map to readable emotion name
        emotion_name = self.emotion_mapping.get(emotion, emotion.title())
        
        return {
            'emotion': emotion_name,
            'confidence': round(confidence, 3),
            'raw_emotion': emotion,
            'method': 'goemotions_model'
        }
    
    def save_model(self, model_path='emotion_model.joblib', vectorizer_path='emotion_vectorizer.joblib'):
        """Save the trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, 'emotion_label_encoder.joblib')
        
        # Save emotion mapping
        with open('emotion_mapping.json', 'w') as f:
            json.dump(self.emotion_mapping, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Label encoder saved to emotion_label_encoder.joblib")
        print("Emotion mapping saved to emotion_mapping.json")
    
    def load_model(self, model_path='emotion_model.joblib', vectorizer_path='emotion_vectorizer.joblib'):
        """Load a pre-trained model and vectorizer"""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load('emotion_label_encoder.joblib')
        
        # Load emotion mapping
        with open('emotion_mapping.json', 'r') as f:
            self.emotion_mapping = json.load(f)
        
        self.emotion_labels = list(self.label_encoder.classes_)
        self.is_trained = True
        
        print("Model, vectorizer, and label encoder loaded successfully")

def main():
    """Main training function"""
    print("Starting emotion analysis model training...")
    
    # Create and train the model
    analyzer = EmotionAnalyzer()
    accuracy = analyzer.train()
    
    # Save the trained model
    analyzer.save_model()
    
    # Test the model
    print("\nTesting the model:")
    test_texts = [
        "I'm so happy today!",
        "This is really sad",
        "I'm angry about this",
        "I love you so much",
        "I'm scared of spiders"
    ]
    
    for text in test_texts:
        result = analyzer.predict_emotion(text)
        print(f"Text: '{text}'")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']})")
        print()

if __name__ == "__main__":
    main()
