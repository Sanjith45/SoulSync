"""
Enhanced Emotion Analysis Module for SoulSync
Improved emotion detection with 6 key emotions and better accuracy
"""

import re
from typing import Dict, List, Tuple
import random

class EnhancedEmotionAnalyzer:
    def __init__(self):
        # Define 6 key emotions with comprehensive keyword sets
        self.emotion_keywords = {
            'joy': {
                'keywords': [
                    'happy', 'joy', 'excited', 'thrilled', 'amazing', 'wonderful', 'great', 'fantastic', 
                    'love', 'adore', 'smile', 'laugh', 'cheerful', 'delighted', 'ecstatic', 'blissful', 
                    'content', 'pleased', 'satisfied', 'glad', 'elated', 'jubilant', 'euphoric', 'radiant', 
                    'sunny', 'bright', 'positive', 'good', 'awesome', 'brilliant', 'excellent', 'perfect', 
                    'beautiful', 'lovely', 'sweet', 'nice', 'cool', 'fun', 'enjoy', 'celebrate', 'success', 
                    'win', 'victory', 'achievement', 'proud', 'accomplished', 'grateful', 'blessed', 'lucky'
                ],
                'weight': 1.0
            },
            'sadness': {
                'keywords': [
                    'sad', 'depressed', 'down', 'lonely', 'crying', 'hurt', 'miserable', 'empty', 
                    'heartbroken', 'devastated', 'grief', 'sorrow', 'melancholy', 'gloomy', 'blue', 
                    'unhappy', 'disappointed', 'dejected', 'despair', 'hopeless', 'lost', 'broken', 
                    'tears', 'weep', 'mourn', 'grieve', 'pain', 'suffering', 'anguish', 'distress', 
                    'unfortunate', 'tragic', 'awful', 'terrible', 'horrible', 'bad', 'worst', 'fail', 
                    'failure', 'defeat', 'loss', 'miss', 'regret', 'guilt', 'shame', 'embarrassed', 
                    'ashamed', 'worthless', 'useless', 'hopeless', 'helpless', 'alone', 'isolated'
                ],
                'weight': 1.0
            },
            'anger': {
                'keywords': [
                    'angry', 'mad', 'furious', 'hate', 'rage', 'outraged', 'livid', 'annoyed', 
                    'frustrated', 'irritated', 'fuming', 'seething', 'enraged', 'incensed', 'infuriated', 
                    'exasperated', 'aggravated', 'bothered', 'upset', 'displeased', 'disgusted', 'repulsed', 
                    'revolted', 'sickened', 'appalled', 'horrified', 'shocked', 'stunned', 'disgusting', 
                    'gross', 'awful', 'terrible', 'horrible', 'hateful', 'vile', 'nasty', 'mean', 'cruel', 
                    'unfair', 'wrong', 'stupid', 'idiot', 'damn', 'hell', 'pissed', 'annoying', 'irritating'
                ],
                'weight': 1.0
            },
            'fear': {
                'keywords': [
                    'scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'frightened', 
                    'panic', 'dread', 'alarmed', 'concerned', 'uneasy', 'troubled', 'restless', 'tense', 
                    'stressed', 'overwhelmed', 'intimidated', 'threatened', 'vulnerable', 'unsafe', 
                    'danger', 'risk', 'threat', 'worry', 'concern', 'anxiety', 'nervousness', 'fearful', 
                    'apprehensive', 'hesitant', 'cautious', 'suspicious', 'paranoid', 'phobia', 'terror', 
                    'horror', 'nightmare', 'frightening', 'scary', 'intimidating', 'daunting', 'overwhelming'
                ],
                'weight': 1.0
            },
            'love': {
                'keywords': [
                    'love', 'adore', 'cherish', 'care', 'special', 'wonderful', 'amazing', 'grateful', 
                    'treasure', 'precious', 'dear', 'beloved', 'sweetheart', 'darling', 'honey', 'baby', 
                    'affection', 'fondness', 'attachment', 'devotion', 'passion', 'romance', 'heart', 
                    'soul', 'beautiful', 'gorgeous', 'stunning', 'perfect', 'angel', 'prince', 'princess', 
                    'soulmate', 'partner', 'family', 'friend', 'support', 'comfort', 'warmth', 'tender', 
                    'gentle', 'kind', 'caring', 'loving', 'appreciate', 'value', 'respect', 'admire'
                ],
                'weight': 1.0
            },
            'surprise': {
                'keywords': [
                    'surprised', 'shocked', 'amazed', 'wow', 'unexpected', 'incredible', 'unbelievable', 
                    'astonished', 'stunned', 'bewildered', 'confused', 'puzzled', 'perplexed', 'baffled', 
                    'mystified', 'startled', 'alarmed', 'sudden', 'abrupt', 'unforeseen', 'unanticipated', 
                    'out of nowhere', 'suddenly', 'all of a sudden', 'really', 'seriously', 'no way', 
                    'can\'t believe', 'mind blown', 'speechless', 'dumbfounded', 'flabbergasted', 'gobsmacked',
                    'unexpected', 'surprising', 'shocking', 'amazing', 'incredible', 'unbelievable'
                ],
                'weight': 1.0
            }
        }
        
        # Emotion emojis
        self.emotion_emojis = {
            'joy': '😊',
            'sadness': '😢', 
            'anger': '😠',
            'fear': '😨',
            'love': '❤️',
            'surprise': '😲'
        }
        
        # Emotion descriptions
        self.emotion_descriptions = {
            'joy': 'Feeling happy and positive',
            'sadness': 'Feeling down or melancholy', 
            'anger': 'Feeling frustrated or upset',
            'fear': 'Feeling anxious or scared',
            'love': 'Feeling affection and care',
            'surprise': 'Feeling astonished or amazed'
        }
    
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
        
        # Analyze using enhanced keyword matching
        return self._enhanced_emotion_detection(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text
    
    def _enhanced_emotion_detection(self, text: str) -> Dict[str, any]:
        """Enhanced emotion detection using weighted keyword matching"""
        words = text.split()
        emotion_scores = {}
        
        # Calculate scores for each emotion
        for emotion, data in self.emotion_keywords.items():
            score = 0
            keywords = data['keywords']
            weight = data['weight']
            
            for word in words:
                if word in keywords:
                    # Give higher score for exact matches
                    score += weight
                    
                    # Check for word variations (e.g., "happier" contains "happy")
                    for keyword in keywords:
                        if keyword in word or word in keyword:
                            score += weight * 0.5
            
            emotion_scores[emotion] = score
        
        # Find the emotion with highest score
        if emotion_scores:
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            
            if max_score > 0:
                # Calculate confidence based on score and text length
                confidence = min(max_score / len(words), 1.0) if words else 0.0
                confidence = max(confidence, 0.3)  # Minimum confidence
                
                return {
                    'emotion': max_emotion.title(),
                    'confidence': round(confidence, 3),
                    'raw_emotion': max_emotion,
                    'method': 'enhanced_keyword_matching'
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
        return self.emotion_emojis.get(emotion.lower(), '😐')
    
    def get_emotion_description(self, emotion: str) -> str:
        """Get a brief description of the emotion"""
        return self.emotion_descriptions.get(emotion.lower(), 'Feeling a mix of emotions')
    
    def get_all_emotions(self) -> List[str]:
        """Get list of all supported emotions"""
        return list(self.emotion_keywords.keys())

# Global instance for easy import
enhanced_emotion_analyzer = EnhancedEmotionAnalyzer()

def analyze_emotion(text: str) -> Dict[str, any]:
    """
    Convenience function to analyze emotion in text
    Returns emotion analysis result
    """
    return enhanced_emotion_analyzer.analyze_emotion(text)

if __name__ == "__main__":
    # Test the enhanced emotion analyzer
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
        "The weather is nice today",
        "I'm feeling confused about everything",
        "I'm proud of my achievements",
        "I feel grateful for your support",
        "I'm disappointed with the results",
        "I'm curious about what happens next",
        "Wow! I can't believe this happened!",
        "I hate this so much, it's terrible",
        "I'm terrified of spiders",
        "You mean everything to me",
        "I'm so excited and thrilled!"
    ]
    
    print("🧠 Testing Enhanced Emotion Analyzer")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        result = analyze_emotion(text)
        emoji = enhanced_emotion_analyzer.get_emotion_emoji(result['emotion'])
        description = enhanced_emotion_analyzer.get_emotion_description(result['emotion'])
        
        print(f"\n{i:2d}. Text: '{text}'")
        print(f"    Emotion: {emoji} {result['emotion']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Method: {result['method']}")
        print(f"    Description: {description}")
        print("-" * 50)
    
    print("\n✅ Enhanced emotion analysis test completed!")
    print("The system now supports 6 key emotions: Joy, Sadness, Anger, Fear, Love, Surprise")
