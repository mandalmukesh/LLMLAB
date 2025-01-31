def simple_classifier(text):
    if 'love' in text or 'fantastic' in text or 'wonderful' in text:
        return "Positive"
    elif 'worst' in text or 'hate' in text or 'terrible' in text:
        return "Negative"
    else:
        return "Neutral"

# Sample sentences to classify
texts = [
    "I love this product!",
    "This is the worst experience I've ever had.",
    "It is okay, not great but not bad either.",
    "Fantastic work!",
    "I hate this.",
    "Absolutely wonderful!",
    "This is terrible."
]

# Classify and print results
for text in texts:
    print(f"Text: {text} - Sentiment: {simple_classifier(text.lower())}")