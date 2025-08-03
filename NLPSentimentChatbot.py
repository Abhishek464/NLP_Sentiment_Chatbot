!pip install transformers

# ✅ Sentiment Analysis using HuggingFace
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

sentences = [
    "I love this product, it’s amazing!",
    "This is the worst service I’ve ever had.",
    "The movie was okay, not great but not terrible."
]

for sentence in sentences:
    result = sentiment_pipeline(sentence)[0]
    print(f"Sentence: {sentence}")
    print(f"Label: {result['label']}, Confidence: {round(result['score'], 3)}\n")

    # ✅ Sentiment Analysis using Naive Bayes (Simulated IMDb)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "I absolutely loved the film!",
    "It was a terrible experience.",
    "An okay performance.",
    "Not my type of movie.",
    "A must-watch!"
]
labels = [1, 0, 1, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.4)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

predictions = model.predict(X_test)
for sentence, pred in zip(X_test, predictions):
    label = "Positive" if pred == 1 else "Negative"
    print(f"Text: {sentence}\nPredicted Sentiment: {label}\n")

# ✅ Simple AI Chatbot using DialoGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

print("Start chatting with the bot (type 'quit' to stop)")
chat_history_ids = None
step = 0

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if step > 0 else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {bot_output}")
    step += 1