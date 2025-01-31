def preprocess(text):
   import re
   import string
   text = text.lower()
   text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
   return text
def tokenize(text):
    return text.split()
# Process the sentences
processed_sentences = [(preprocess(s[0]), s[1]) for s in sentences]
tokens = [tokenize(s[0]) for s in processed_sentences]
# Print the results
print("Processed Sentences:")
for ps in processed_sentences:
  print(ps)
print("\nTokens:")
for tk in tokens:
   print(tk)