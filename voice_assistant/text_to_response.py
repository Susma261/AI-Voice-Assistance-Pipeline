#text_to_response.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def generate_response(query):
    """Generates a relevant response using a pre-trained model from Hugging Face."""
    try:
        # Using a smaller model like distilgpt2
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Use the text-generation pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Generate a response using the model
        prompt = f"Q: {query}\nA:"

        response = generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            temperature=0.5,
            num_beams=5,
            truncation=True  # Explicitly set truncation
        )[0]['generated_text']

        # Keep only the relevant part by truncating after the first two sentences
        sentences = response.split('. ')
        if len(sentences) > 1:
            response = '. '.join(sentences[:2]).strip() + ('.' if response.endswith('.') else '')
        else:
            response = response.strip()

        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
