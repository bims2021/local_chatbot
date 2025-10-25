# Local Chatbot with Hugging Face

A fully functional command-line chatbot that runs locally using Hugging Face text generation models with conversational memory.

## Features

- Local text generation using Hugging Face models
- Sliding window conversation memory
- Simple command-line interface
- Modular and extensible codebase
- GPU support (optional)

**Project Structure**:

    model_loader.py - Handles model loading and text generation

    chat_memory.py - Manages conversation history with sliding window

    interface.py - Main CLI interface and chat loop

    requirements.txt - Python dependencies

    README.md-Task Documentaion

## Setup

1. **Clone or download the project files**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate

**Install dependencies**
   pip install -r requirements.txt

**Run the chatbot**:
   python interface.py 

**Available commands in chat**:

    /exit - Exit the chatbot

    /clear - Clear conversation memory

    /help - Show help message

**Model Options**:
The default model is microsoft/DialoGPT-small. You can modify the model in interface.py:         
chatbot = ChatbotInterface(
    model_name="microsoft/DialoGPT-small",  # Change this
    memory_window=3
)

**Other small models to try**:

    microsoft/DialoGPT-medium

    gpt2

    distilgpt2


**Example Interaction**:
User: Hello, how are you?
Bot: I'm doing well, thank you! How are you today?

User: I'm good too. Can you tell me about AI?
Bot: Artificial Intelligence is a field of computer science that focuses on creating machines that can perform tasks that typically require human intelligence.

User: What are some applications?
Bot: Some common applications include natural language processing, computer vision, robotics, and machine learning systems.

User: /exit
Exiting chatbot. Goodbye!

**Memory Management**:

The chatbot maintains a sliding window of the last 3-5 conversation exchanges (configurable). This ensures:

    Coherent multi-turn conversations

    Limited memory usage

    Context-aware responses

**Troubleshooting**:

    Out of memory errors: Try a smaller model like distilgpt2

    Slow responses: The first run may be slower as models are downloaded

    Model not found: Ensure you have an internet connection for first-time downloads