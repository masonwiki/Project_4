from chat_bot import ChatBot

def main():
    # Initialize the chatbot (e.g., load the model)
    chatbot = ChatBot()

    print("Hello! I'm your chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Use the chatbot logic to process the user input and generate a response
        response = chatbot.generate_response(user_input)

        print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()