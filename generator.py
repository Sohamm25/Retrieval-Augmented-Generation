from transformers import pipeline
# Initialize the generator pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-small")
query = "Where is the Eiffel Tower located?"
context = "The Eiffel Tower is in Paris."
# Generate answer
input_text = f"Answer the question: {query} Context: {context}"
response = generator(input_text)
print(f"Generated Answer: {response[0]['generated_text']}")
