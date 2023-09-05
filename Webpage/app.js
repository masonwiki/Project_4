// Creates and stores a conversation history
let conversationHistory = [];
function addToHistory(message) {
  conversationHistory.push(message);
}

// Displays the conversation in the chat container
function displayConversation(index) {
  const chatMessages = document.getElementById('chat-messages');
  chatMessages.innerHTML = conversationHistory[index];
}

function loadConversationButtons() {
  const conversationList = document.getElementById('conversation-list');

  for (let i = 0; i < conversationHistory.length; i++) {
    const listItem = document.createElement('li');
    const button = document.createElement('button');
    button.textContent = `Conversation ${i + 1}`;
    button.addEventListener('click', () => {
      displayConversation(i);
    });
    listItem.appendChild(button);
    conversationList.appendChild(listItem);
  }
}

const sendButton = document.getElementById('send-button');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', function(event) {
  if (event.key === 'Enter') {
    sendMessage();
  }
});

function sendMessage() {
  const userMessage = userInput.value;
  const userMessageElement = document.createElement('div');

  userMessageElement.classList.add('message', 'user-message');
  userMessageElement.textContent = `User: ${userMessage}`;

  chatMessages.appendChild(userMessageElement);
  addToHistory(userMessageElement.outerHTML);

  userInput.value = '';
}
loadConversationButtons();
