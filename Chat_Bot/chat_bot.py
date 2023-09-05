import random
import json
import torch
from bot_training.model import NeuralNet
from bot_training.nltk_utils import bag_of_words, tokenize

class ChatBot:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open('Resources\indents.json', 'r') as f:
            self.intents = json.load(f)

        file = 'Resources\data.pth'
        data = torch.load(file)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        model_state = data["model_state"]

        self.loaded_model = NeuralNet(input_size, hidden_size, output_size).to(device)
        self.loaded_model.load_state_dict(model_state)
        self.loaded_model.eval()

    def generate_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)  # Use self.all_words
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.loaded_model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]  # Use self.tags

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])
        else:
            return "I do not understand your question. Rephrase it and try again."

if __name__ == '__main__':
    chatbot = ChatBot()
    bot_name = "Stock Bot"
    print("Hello! I'm Stock Bot! Ask me any stock-related question. Type 'quit' to exit.")
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break

        response = chatbot.generate_response(sentence)
        print(f'{bot_name}: {response}')