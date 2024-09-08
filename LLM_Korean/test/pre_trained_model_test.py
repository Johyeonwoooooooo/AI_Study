from transformers import pipeline

messages = ["how to write good report"]

print("Model build...")
pipe = pipeline("text-generation", model="beomi/Llama-3-KoEn-8B")
print("Model build complete")

print("Generating text...")
result = pipe(messages, max_length=100, truncation=True)

print("Result:")
print(result)
