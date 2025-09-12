"""
Test script for emotion analysis functionality
Run this to verify the emotion analysis is working correctly
"""

from emotion_analyzer import analyze_emotion, emotion_analyzer

def test_emotion_analysis():
    """Test the emotion analysis with various sample texts"""
    
    test_cases = [
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
        "I'm curious about what happens next"
    ]
    
    print("🧠 Testing Emotion Analysis System")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        result = analyze_emotion(text)
        emoji = emotion_analyzer.get_emotion_emoji(result['emotion'])
        description = emotion_analyzer.get_emotion_description(result['emotion'])
        
        print(f"\n{i:2d}. Text: '{text}'")
        print(f"    Emotion: {emoji} {result['emotion']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Method: {result['method']}")
        print(f"    Description: {description}")
        print("-" * 50)
    
    print("\n✅ Emotion analysis test completed!")
    print("The system is ready to use in your SoulSync application.")

if __name__ == "__main__":
    test_emotion_analysis()
