"""
BERT-based Emotion Analysis Module for SoulSync
High-accuracy emotion detection using pre-trained BERT models
Supports multi-label emotion prediction with contextual understanding
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Tuple, Union
import json
import os
import re
from datetime import datetime

class BERTEmotionAnalyzer:
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize BERT-based emotion analyzer
        
        Args:
            model_name: Hugging Face model name for emotion classification
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Emotion mapping for GoEmotions compatibility
        self.emotion_mapping = {
            'admiration': 'Admiration',
            'amusement': 'Amusement', 
            'anger': 'Anger',
            'annoyance': 'Annoyance',
            'approval': 'Approval',
            'caring': 'Caring',
            'confusion': 'Confusion',
            'curiosity': 'Curiosity',
            'desire': 'Desire',
            'disappointment': 'Disappointment',
            'disapproval': 'Disapproval',
            'disgust': 'Disgust',
            'embarrassment': 'Embarrassment',
            'excitement': 'Excitement',
            'fear': 'Fear',
            'gratitude': 'Gratitude',
            'grief': 'Grief',
            'joy': 'Joy',
            'love': 'Love',
            'nervousness': 'Nervousness',
            'optimism': 'Optimism',
            'pride': 'Pride',
            'realization': 'Realization',
            'relief': 'Relief',
            'remorse': 'Remorse',
            'sadness': 'Sadness',
            'surprise': 'Surprise',
            'neutral': 'Neutral'
        }
        
        # Emotion emojis
        self.emotion_emojis = {
            'admiration': '🤩', 'amusement': '😄', 'anger': '😠', 'annoyance': '😤',
            'approval': '👍', 'caring': '🤗', 'confusion': '😕', 'curiosity': '🤔',
            'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
            'embarrassment': '😳', 'excitement': '🤩', 'fear': '😨', 'gratitude': '🙏',
            'grief': '😭', 'joy': '😊', 'love': '❤️', 'nervousness': '😬',
            'optimism': '😌', 'pride': '😌', 'realization': '💡', 'relief': '😌',
            'remorse': '😔', 'sadness': '😢', 'surprise': '😲', 'neutral': '😐'
        }
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the BERT emotion classification model"""
        try:
            print(f"Loading BERT emotion model: {self.model_name}")
            
            # Use pipeline for easy inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,  # Return all scores
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.is_loaded = True
            print("BERT emotion model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Falling back to keyword-based emotion detection")
            self.is_loaded = False
    
    def analyze_emotion(self, text: str, multi_label: bool = True, threshold: float = 0.3) -> Dict:
        """
        Analyze emotion in text using BERT
        
        Args:
            text: Input text to analyze
            multi_label: Whether to return multiple emotions (True) or single top emotion (False)
            threshold: Confidence threshold for multi-label predictions
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not text or not text.strip():
            return {
                'emotion': 'Neutral',
                'confidence': 0.0,
                'raw_emotion': 'neutral',
                'method': 'empty_text',
                'emotions': [{'emotion': 'Neutral', 'confidence': 0.0}]
            }
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        if self.is_loaded:
            try:
                return self._predict_with_bert(cleaned_text, multi_label, threshold)
            except Exception as e:
                print(f"BERT prediction failed: {e}")
                return self._fallback_emotion_detection(cleaned_text)
        else:
            return self._fallback_emotion_detection(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Keep original case for BERT (it's case-sensitive)
        return text
    
    def _predict_with_bert(self, text: str, multi_label: bool, threshold: float) -> Dict:
        """Predict emotions using BERT model"""
        try:
            # Get predictions from BERT pipeline
            results = self.pipeline(text)
            
            # Handle the nested list structure returned by the pipeline
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    results = results[0]  # Extract the inner list
                elif isinstance(results[0], dict):
                    results = results  # Already in correct format
            
            emotions = []
            for result in results:
                emotion_name = result['label'].lower()
                confidence = result['score']
                
                # Map to our emotion format
                mapped_emotion = self.emotion_mapping.get(emotion_name, emotion_name.title())
                
                emotions.append({
                    'emotion': mapped_emotion,
                    'confidence': round(confidence, 3),
                    'raw_emotion': emotion_name
                })
            
            # Sort by confidence
            emotions.sort(key=lambda x: x['confidence'], reverse=True)
            
            if multi_label:
                # Return multiple emotions above threshold
                top_emotions = [e for e in emotions if e['confidence'] >= threshold]
                primary_emotion = emotions[0] if emotions else {'emotion': 'Neutral', 'confidence': 0.0}
                
                return {
                    'emotion': primary_emotion['emotion'],
                    'confidence': primary_emotion['confidence'],
                    'raw_emotion': primary_emotion.get('raw_emotion', primary_emotion['emotion'].lower()),
                    'method': 'bert_model',
                    'emotions': top_emotions,
                    'multi_label': True
                }
            else:
                # Return single top emotion
                primary_emotion = emotions[0] if emotions else {'emotion': 'Neutral', 'confidence': 0.0}
                
                return {
                    'emotion': primary_emotion['emotion'],
                    'confidence': primary_emotion['confidence'],
                    'raw_emotion': primary_emotion.get('raw_emotion', primary_emotion['emotion'].lower()),
                    'method': 'bert_model',
                    'emotions': [primary_emotion],
                    'multi_label': False
                }
                
        except Exception as e:
            print(f"Error in BERT prediction: {e}")
            return self._fallback_emotion_detection(text)
    
    def _fallback_emotion_detection(self, text: str) -> Dict:
        """Fallback emotion detection using keyword matching"""
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'amazing', 'wonderful', 'great', 'fantastic'],
            'sadness': ['sad', 'depressed', 'down', 'lonely', 'crying', 'hurt', 'miserable'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'rage', 'outraged', 'livid'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
            'love': ['love', 'adore', 'cherish', 'care', 'special', 'wonderful'],
            'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'unexpected']
        }
        
        words = text.lower().split()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for word in words if word in keywords)
            emotion_scores[emotion] = score
        
        if emotion_scores:
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            
            if max_score > 0:
                confidence = min(max_score / len(words), 1.0) if words else 0.0
                return {
                    'emotion': max_emotion.title(),
                    'confidence': round(confidence, 3),
                    'raw_emotion': max_emotion,
                    'method': 'keyword_fallback',
                    'emotions': [{'emotion': max_emotion.title(), 'confidence': confidence}],
                    'multi_label': False
                }
        
        return {
            'emotion': 'Neutral',
            'confidence': 0.5,
            'raw_emotion': 'neutral',
            'method': 'keyword_fallback',
            'emotions': [{'emotion': 'Neutral', 'confidence': 0.5}],
            'multi_label': False
        }
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji representation for emotion"""
        return self.emotion_emojis.get(emotion.lower(), '😐')
    
    def get_emotion_description(self, emotion: str) -> str:
        """Get a brief description of the emotion"""
        descriptions = {
            'admiration': 'Feeling respect and esteem',
            'amusement': 'Feeling entertained and amused',
            'anger': 'Feeling frustrated or upset',
            'annoyance': 'Feeling irritated or bothered',
            'approval': 'Feeling supportive and agreeing',
            'caring': 'Feeling compassionate and concerned',
            'confusion': 'Feeling puzzled or uncertain',
            'curiosity': 'Feeling interested and inquisitive',
            'desire': 'Feeling longing or wanting',
            'disappointment': 'Feeling let down or discouraged',
            'disapproval': 'Feeling opposed or disagreeing',
            'disgust': 'Feeling repulsed or revolted',
            'embarrassment': 'Feeling self-conscious or ashamed',
            'excitement': 'Feeling enthusiastic and energetic',
            'fear': 'Feeling anxious or scared',
            'gratitude': 'Feeling thankful and appreciative',
            'grief': 'Feeling deep sorrow or loss',
            'joy': 'Feeling happy and positive',
            'love': 'Feeling affection and care',
            'nervousness': 'Feeling uneasy or apprehensive',
            'optimism': 'Feeling hopeful and positive',
            'pride': 'Feeling accomplished and satisfied',
            'realization': 'Feeling understanding or insight',
            'relief': 'Feeling comforted and reassured',
            'remorse': 'Feeling regretful or guilty',
            'sadness': 'Feeling down or melancholy',
            'surprise': 'Feeling astonished or amazed',
            'neutral': 'Feeling calm and balanced'
        }
        
        return descriptions.get(emotion.lower(), 'Feeling a mix of emotions')
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'supports_multi_label': True,
            'emotion_count': len(self.emotion_mapping)
        }

# Global instance for easy import
bert_emotion_analyzer = BERTEmotionAnalyzer()

def analyze_emotion_bert(text: str, multi_label: bool = True, threshold: float = 0.3) -> Dict:
    """
    Convenience function to analyze emotion using BERT
    """
    return bert_emotion_analyzer.analyze_emotion(text, multi_label, threshold)

if __name__ == "__main__":
    # Test the BERT emotion analyzer
    test_texts = [
        "I'm so happy and excited about the new job!",
        "I'm feeling sad and disappointed about the results",
        "I'm angry and frustrated with this situation",
        "I love you so much and I'm grateful for everything",
        "I'm scared and nervous about the presentation",
        "Wow! I'm surprised and amazed by this news!",
        "I'm feeling confused and uncertain about what to do",
        "I'm proud and optimistic about the future"
    ]
    
    print("🧠 Testing BERT Emotion Analyzer")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")
        
        # Single label prediction
        result_single = analyze_emotion_bert(text, multi_label=False)
        emoji = bert_emotion_analyzer.get_emotion_emoji(result_single['emotion'])
        print(f"   Single: {emoji} {result_single['emotion']} (confidence: {result_single['confidence']:.3f})")
        
        # Multi-label prediction
        result_multi = analyze_emotion_bert(text, multi_label=True, threshold=0.2)
        print(f"   Multi:  ", end="")
        for emotion_data in result_multi['emotions'][:3]:  # Show top 3
            emoji = bert_emotion_analyzer.get_emotion_emoji(emotion_data['emotion'])
            print(f"{emoji} {emotion_data['emotion']}({emotion_data['confidence']:.2f}) ", end="")
        print()
        print("-" * 50)
    
    print("\n✅ BERT emotion analysis test completed!")
    print("Model info:", bert_emotion_analyzer.get_model_info())
