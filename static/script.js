const chatbotIcon = document.getElementById('chatbot-icon');
const chatContainer = document.getElementById('chat-container');
const closeChat = document.getElementById('close-chat');
const form = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');
const messagesList = document.getElementById('messages');

chatbotIcon.addEventListener('click', function() {
    if (chatContainer.style.display === 'none' || chatContainer.style.display === '') {
        chatContainer.style.display = 'flex';
    } else {
        chatContainer.style.display = 'none';
    }
});

closeChat.addEventListener('click', function() {
    chatContainer.style.display = 'none';
});

form.addEventListener('submit', function(event) {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (message === '') return;
    appendMessage('user', message);
    sendMessage(message);
    messageInput.value = '';
});

function appendMessage(sender, message) {
    const li = document.createElement('li');
    li.textContent = message;
    li.classList.add('message', sender + '-message');
    messagesList.appendChild(li);
    messagesList.scrollTop = messagesList.scrollHeight;
}

function sendMessage(message) {
    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        appendMessage('bot', data.response);
    })
    .catch(error => console.error('Erreur lors de l\'envoi du message:', error));
}

// Message de bienvenue
setTimeout(() => {
    appendMessage('bot', 'Bonjour ! Je suis votre assistant de voyage. Comment puis-je vous aider aujourd\'hui ?');
}, 1000);

