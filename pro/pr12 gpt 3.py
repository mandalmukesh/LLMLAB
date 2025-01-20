prompts = ["Tell a story about a robot.", "Describe a futuristic city.", "Write a poemabout love."]
for prompt in prompts:
    result = generator(prompt, max_length=50, num_return_sequences=1) # Indented this line
    print(result[0]['generated_text']) # Indented this line