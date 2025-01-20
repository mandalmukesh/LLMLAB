from transformers import pipeline
# Load a text generation model
generator = pipeline("text-generation", model="gpt2")
# Define the prompt and generate text
prompt = "Once there was a dragon"
result = generator(prompt, max_length=50, num_return_sequences=1)
# Print the generated text
print(result[0]['generated_text'])