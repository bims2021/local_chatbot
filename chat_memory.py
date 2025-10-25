from collections import deque

class ChatMemory:
    def __init__(self, window_size=4):
        """
        Maintain conversation history using a sliding window.
        
        Args:
            window_size (int): Number of conversation turns to remember
        """
        self.window_size = window_size
        # Store as list of exchanges (each exchange is user + bot)
        self.conversation_history = deque(maxlen=window_size)
    
    def add_exchange(self, user_message, bot_response):
        """
        Add a user-bot exchange to the conversation history.
        
        Args:
            user_message (str): User's input message
            bot_response (str): Bot's response
        """
        self.conversation_history.append({
            'user': user_message,
            'bot': bot_response
        })
    
    def get_context(self):
        """
        Get the recent conversation context as a formatted string.
        For DialoGPT, format as natural conversation flow.
        
        Returns:
            str: Formatted conversation context
        """
        if not self.conversation_history:
            return ""
        
        context_lines = []
        for exchange in self.conversation_history:
            context_lines.append(f"User: {exchange['user']}")
            context_lines.append(f"Bot: {exchange['bot']}")
        
        return "\n".join(context_lines) + "\n"
    
    def clear_memory(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
    
    def get_history_length(self):
        """Get the number of exchanges in memory."""
        return len(self.conversation_history)
    
    def __str__(self):
        """String representation of conversation history."""
        return self.get_context()