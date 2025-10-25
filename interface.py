import os
from model_loader import ModelLoader
from chat_memory import ChatMemory

class ChatbotInterface:
    def __init__(self, model_name="microsoft/DialoGPT-medium", memory_window=4):
        """
        Initialize the chatbot interface.
        
        Args:
            model_name (str): Hugging Face model identifier
            memory_window (int): Size of conversation memory window
        """
        self.model_loader = ModelLoader(model_name)
        self.memory = ChatMemory(memory_window)
        self.is_running = False
        
    def setup(self):
        """Setup the chatbot by loading the model."""
        print("Initializing Local Chatbot...")
        print("=" * 40)
        
        if not self.model_loader.load_model():
            print("Failed to load model. Exiting.")
            return False
            
        print(f"Memory window: {self.memory.window_size} exchanges")
        print("Type '/exit' to quit, '/clear' to clear memory")
        print("=" * 40)
        return True
    
    def process_user_input(self, user_input):
        """
        Process user input and generate bot response.
        
        Args:
            user_input (str): User's input message
            
        Returns:
            tuple: (should_continue, bot_response)
        """
        # Handle special commands
        if user_input.lower() == '/exit':
            return False, "Exiting chatbot. Goodbye!"
            
        elif user_input.lower() == '/clear':
            self.memory.clear_memory()
            return True, "Conversation memory cleared."
        
        # Generate response for normal input
        try:
            # Build prompt with conversation context
            context = self.memory.get_context()
            # Add a system-like instruction for better responses
            if not context:
                full_prompt = f"User: {user_input}\nBot:"
            else:
                full_prompt = f"{context}User: {user_input}\nBot:"
            
            # Generate response using pipeline
            bot_response = self.model_loader.generate_response(
                full_prompt,
                max_new_tokens=50,
                temperature=0.6,  # Lower for more focused responses
                top_p=0.85,       # Nucleus sampling for better quality
                repetition_penalty=1.15,
                do_sample=True,
            )
            
            # Add exchange to memory
            self.memory.add_exchange(user_input, bot_response)
            
            return True, bot_response
            
        except Exception as e:
            return True, f"Error: {str(e)}"
    
    def run(self):
        """Main chat loop."""
        if not self.setup():
            return
        
        self.is_running = True
        print("\nChatbot is ready! Start chatting...\n")
        
        while self.is_running:
            try:
                # Get user input
                user_input = input("User: ").strip()
                
                if not user_input:
                    continue
                
                # Process input
                should_continue, response = self.process_user_input(user_input)
                
                # Display response
                if should_continue:
                    print(f"Bot: {response}\n")
                else:
                    print(f"\n{response}")
                    self.is_running = False
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                self.is_running = False
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                self.is_running = False

def main():
    """Main function to start the chatbot."""
    # You can change the model name here for different models:
    # - "microsoft/DialoGPT-small"   (fastest)
    # - "microsoft/DialoGPT-medium"  (recommended)
    # - "microsoft/DialoGPT-large"   (best quality, slower)
    # - "gpt2"                       (alternative)
    # - "distilgpt2"                 (fast alternative)
    
    chatbot = ChatbotInterface(
        model_name="microsoft/DialoGPT-medium",
        memory_window=4  # Last 4 exchanges
    )
    chatbot.run()

if __name__ == "__main__":
    main()