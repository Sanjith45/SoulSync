"""
Compare old TF-IDF vs new BERT emotion analyzers
"""

from emotion_analyzer import analyze_emotion as analyze_old
from emotion_analyzer_bert import analyze_emotion_bert as analyze_new

def compare_analyzers():
    """Compare accuracy between old and new analyzers"""
    
    test_cases = [
        ("I'm so happy and excited!", "joy"),
        ("I'm feeling really sad and depressed", "sadness"),
        ("I'm angry and frustrated with this", "anger"),
        ("I'm scared and nervous about tomorrow", "fear"),
        ("I love you so much", "love"),
        ("Wow! I'm surprised and amazed!", "surprise"),
        ("Thank you so much for everything", "gratitude"),
        ("I'm disappointed and let down", "disappointment"),
        ("I'm confused and uncertain", "confusion"),
        ("I'm proud and optimistic", "pride")
    ]
    
    print("🔬 Comparing Emotion Analyzers")
    print("=" * 80)
    print(f"{'Text':<40} {'Expected':<15} {'Old Result':<15} {'New Result':<15} {'Improvement'}")
    print("-" * 80)
    
    old_correct = 0
    new_correct = 0
    
    for text, expected in test_cases:
        # Test old analyzer
        old_result = analyze_old(text)
        old_emotion = old_result['emotion'].lower()
        old_conf = old_result['confidence']
        
        # Test new BERT analyzer
        new_result = analyze_new(text, multi_label=False)
        new_emotion = new_result['emotion'].lower()
        new_conf = new_result['confidence']
        
        # Check correctness
        old_is_correct = expected in old_emotion or old_emotion in expected
        new_is_correct = expected in new_emotion or new_emotion in expected
        
        if old_is_correct:
            old_correct += 1
        if new_is_correct:
            new_correct += 1
        
        improvement = "✅" if new_is_correct and not old_is_correct else "➡️" if new_is_correct == old_is_correct else "❌" if not new_is_correct and old_is_correct else "✅"
        
        print(f"{text[:37]:<40} {expected:<15} {old_emotion[:12]:<15} {new_emotion[:12]:<15} {improvement}")
        print(f"{'':40} {'':15} f{old_conf:.2f}        f{new_conf:.2f}")
        print()
    
    old_accuracy = (old_correct / len(test_cases)) * 100
    new_accuracy = (new_correct / len(test_cases)) * 100
    
    print("=" * 80)
    print(f"Old TF-IDF Accuracy: {old_accuracy:.1f}% ({old_correct}/{len(test_cases)})")
    print(f"New BERT Accuracy:   {new_accuracy:.1f}% ({new_correct}/{len(test_cases)})")
    print(f"Improvement:         +{new_accuracy - old_accuracy:.1f} percentage points")
    print("=" * 80)
    
    if new_accuracy > old_accuracy:
        print("🎉 BERT analyzer shows significant improvement!")
    elif new_accuracy == old_accuracy:
        print("➡️ Both analyzers perform equally on this test set")
    else:
        print("⚠️ BERT analyzer needs more testing")

if __name__ == "__main__":
    compare_analyzers()
