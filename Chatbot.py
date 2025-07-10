import json
import random
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

with open("Intent.json") as file:
    data = json.load(file)

intents = data["intents"]  # ✅ get the list of intent blocks

print(json.dumps(data["intents"][:2], indent=2))

texts = []
labels = []

for intent in intents:
    tag = intent.get("intent")
    phrases = intent.get("text")

    if tag and isinstance(phrases, list):
        for sentence in phrases:
            texts.append(sentence.strip())
            labels.append(tag.strip())

df = pd.DataFrame({'text': texts, 'label': labels})
df.head()

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

model.fit(texts, labels)

print("texts length:", len(texts))
print("labels length:", len(labels))
print("Sample text:", texts[:3])
print("Sample labels:", labels[:3])


print("Type of data:", type(data))

responses = {intent['intent']: intent['responses'] for intent in intents}


def chat():
    print("🤖 ChatBot ML is ready! Type 'quit' to exit.")
    while True:
        msg = input("You: ").lower().strip()
        if msg == "quit":
            print("Bot: Bye! 👋")
            break

        predicted_tag = model.predict([msg])[0]
        bot_reply = random.choice(responses.get(predicted_tag, ["Hmm... I don’t understand that."]))
        print("Bot:", bot_reply)

chat()







