"""
Emotion Analysis Module for SoulSync
Replaces basic sentiment analysis with detailed emotion classification
"""

import joblib
import json
import os
import re
from typing import Dict, List, Tuple

class EmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.emotion_mapping = {}
        self.is_loaded = False
        
        # Load the model if it exists
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained emotion model"""
        try:
            if (os.path.exists('emotion_model.joblib') and 
                os.path.exists('emotion_vectorizer.joblib') and
                os.path.exists('emotion_label_encoder.joblib')):
                
                self.model = joblib.load('emotion_model.joblib')
                self.vectorizer = joblib.load('emotion_vectorizer.joblib')
                self.label_encoder = joblib.load('emotion_label_encoder.joblib')
                
                # Load emotion mapping
                if os.path.exists('emotion_mapping.json'):
                    with open('emotion_mapping.json', 'r') as f:
                        self.emotion_mapping = json.load(f)
                
                self.is_loaded = True
                print("GoEmotions-trained emotion model loaded successfully")
            else:
                print("Emotion model files not found. Using fallback emotion detection.")
                print("Run 'python train_emotion_model.py' to train the model first.")
                self.is_loaded = False
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.is_loaded = False
    
    def analyze_emotion(self, text: str) -> Dict[str, any]:
        """
        Analyze emotion in the given text
        Returns a dictionary with emotion, confidence, and additional info
        """
        if not text or not text.strip():
            return {
                'emotion': 'Neutral',
                'confidence': 0.0,
                'raw_emotion': 'neutral',
                'method': 'empty_text'
            }
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        if self.is_loaded:
            try:
                return self._predict_with_model(cleaned_text)
            except Exception as e:
                print(f"Model prediction failed: {e}")
                return self._fallback_emotion_detection(cleaned_text)
        else:
            return self._fallback_emotion_detection(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text
    
    def _predict_with_model(self, text: str) -> Dict[str, any]:
        """Predict emotion using the trained model"""
        # Vectorize the text
        text_vec = self.vectorizer.transform([text])
        
        # Get prediction
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
    
    def _fallback_emotion_detection(self, text: str) -> Dict[str, any]:
        """Fallback emotion detection using keyword matching"""
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'amazing', 'wonderful', 'great', 'fantastic', 'love', 'adore', 'smile', 'laugh'],
            'sadness': ['sad', 'depressed', 'down', 'lonely', 'crying', 'hurt', 'miserable', 'empty', 'heartbroken', 'devastated', 'grief'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'rage', 'outraged', 'livid', 'annoyed', 'frustrated', 'irritated'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'frightened', 'panic', 'dread'],
            'love': ['love', 'adore', 'cherish', 'care', 'special', 'wonderful', 'amazing', 'grateful'],
            'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'unexpected', 'incredible', 'unbelievable'],
            'disgust': ['disgusting', 'gross', 'revolting', 'sickening', 'awful', 'terrible', 'horrible'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'blessed', 'fortunate', 'lucky'],
            'anxiety': ['anxious', 'worried', 'stressed', 'nervous', 'concerned', 'uneasy', 'troubled', 'restless'],
            'excitement': ['excited', 'thrilled', 'pumped', 'hyped', 'stoked', 'ecstatic', 'enthusiastic']
        }
        
        # Count emotion indicators
        emotion_scores = {}
        words = text.split()
        
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for word in words:
                if word in keywords:
                    score += 1
            emotion_scores[emotion] = score
        
        # Find the emotion with highest score
        if emotion_scores:
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            
            if max_score > 0:
                # Calculate confidence based on score and text length
                confidence = min(max_score / len(words), 1.0) if words else 0.0
                return {
                    'emotion': max_emotion.title(),
                    'confidence': round(confidence, 3),
                    'raw_emotion': max_emotion,
                    'method': 'keyword_matching'
                }
        
        # Default to neutral if no emotions detected
        return {
            'emotion': 'Neutral',
            'confidence': 0.5,
            'raw_emotion': 'neutral',
            'method': 'default_neutral'
        }
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji representation for emotion"""
        emotion_emojis = {
            'joy': '😊',
            'happiness': '😊',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'love': '❤️',
            'surprise': '😲',
            'disgust': '🤢',
            'gratitude': '🙏',
            'anxiety': '😰',
            'excitement': '🤩',
            'neutral': '😐',
            'caring': '🤗',
            'confusion': '😕',
            'curiosity': '🤔',
            'disappointment': '😞',
            'embarrassment': '😳',
            'grief': '😭',
            'nervousness': '😬',
            'optimism': '😌',
            'pride': '😌',
            'relief': '😌',
            'remorse': '😔'
        }
        
        return emotion_emojis.get(emotion.lower(), '😐')
    
    def get_emotion_description(self, emotion: str) -> str:
        """Get a brief description of the emotion"""
        descriptions = {
            'joy': 'Feeling happy and positive',
            'sadness': 'Feeling down or melancholy',
            'anger': 'Feeling frustrated or upset',
            'fear': 'Feeling anxious or scared',
            'love': 'Feeling affection and care',
            'surprise': 'Feeling astonished or amazed',
            'disgust': 'Feeling repulsed or revolted',
            'gratitude': 'Feeling thankful and appreciative',
            'anxiety': 'Feeling worried or nervous',
            'excitement': 'Feeling enthusiastic and energetic',
            'neutral': 'Feeling calm and balanced',
            'caring': 'Feeling compassionate and concerned',
            'confusion': 'Feeling puzzled or uncertain',
            'curiosity': 'Feeling interested and inquisitive',
            'disappointment': 'Feeling let down or discouraged',
            'embarrassment': 'Feeling self-conscious or ashamed',
            'grief': 'Feeling deep sorrow or loss',
            'nervousness': 'Feeling uneasy or apprehensive',
            'optimism': 'Feeling hopeful and positive',
            'pride': 'Feeling accomplished and satisfied',
            'relief': 'Feeling comforted and reassured',
            'remorse': 'Feeling regretful or guilty'
        }
        
        return descriptions.get(emotion.lower(), 'Feeling a mix of emotions')

# Global instance for easy import
emotion_analyzer = EmotionAnalyzer()

def analyze_emotion(text: str) -> Dict[str, any]:
    """
    Convenience function to analyze emotion in text
    Returns emotion analysis result
    """
    return emotion_analyzer.analyze_emotion(text)

if __name__ == "__main__":
    # Test the emotion analyzer
    test_texts = [
        "I'm so happy today!",
        "This is really sad and depressing",
        "I'm angry about this situation",
        "I love you so much",
        "I'm scared of what might happen",
        "Thank you so much for your help",
        "I'm excited about the upcoming event",
        "This is disgusting and awful",
        "I'm feeling anxious about the exam",
        "The weather is nice today"
    ]
    
    print("Testing Emotion Analyzer:")
    print("=" * 50)
    
    for text in test_texts:
        result = analyze_emotion(text)
        emoji = emotion_analyzer.get_emotion_emoji(result['emotion'])
        description = emotion_analyzer.get_emotion_description(result['emotion'])
        
        print(f"Text: '{text}'")
        print(f"Emotion: {emoji} {result['emotion']} (confidence: {result['confidence']})")
        print(f"Description: {description}")
        print(f"Method: {result['method']}")
        print("-" * 30)
