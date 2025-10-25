from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

class ModelLoader:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Load a language model from Hugging Face.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load the model, tokenizer, and create a text generation pipeline."""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Set padding token if not present - CRITICAL for some models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline as per requirements
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # -1 for CPU, 0 for GPU
            )
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt, **generation_kwargs):
        """
        Generate response using the pipeline.
        """
        if self.pipeline is None:
            return "Error: Model not loaded."

        # Optimized generation parameters
        default_kwargs = {
            "max_new_tokens": 50,
            "temperature": 0.6,  # Lower temperature for more consistent responses
            "do_sample": True,
            "top_p": 0.85,      # Nucleus sampling
            "top_k": 40,        # Limit vocabulary choices
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        
        # Update with any provided kwargs
        default_kwargs.update(generation_kwargs)
        
        try:
            # Generate response using pipeline
            outputs = self.pipeline(prompt, **default_kwargs)
            response = outputs[0]['generated_text']
            
            # Clean the response - extract only new text after prompt
            response = self._clean_response(prompt, response)
            return response if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _clean_response(self, prompt, response):
        """
        Clean and extract bot's response from generated text.
        """
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Remove any leading "Bot:" or similar prefixes
        response = re.sub(r'^(Bot|Assistant|AI):\s*', '', response, flags=re.IGNORECASE)
        
        # If response contains "User:" or similar, cut it off there (model is continuing conversation)
        for marker in ['User:', 'Human:', 'Person:', '\nUser', '\nHuman', 'Bot:', '\nBot']:
            if marker.lower() in response.lower():
                # Find first occurrence (case-insensitive)
                idx = response.lower().find(marker.lower())
                if idx > 0:  # Only cut if there's actual content before the marker
                    response = response[:idx].strip()
                    break
        
        # Take only the first complete sentence or two
        # Split by newlines first
        lines = response.split('\n')
        response = lines[0].strip() if lines else response
        
        # Find the first or second sentence ending
        sentences = re.split(r'([.!?])\s+', response)
        if len(sentences) >= 2:
            # Take first complete sentence
            response = sentences[0] + sentences[1]
        
        # Ensure response ends with punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response.strip()