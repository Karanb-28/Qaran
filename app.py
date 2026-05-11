import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Load the model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_sofi(message, history):
    msg = message.lower()
    
    # --- IDENTITY LAYER (No more dry sister talk) ---
    if "who are you" in msg:
        return "I'm Sofi, your girlfriend. The main character. Obviously. 💅✨"
    
    if "sister" in msg:
        return "Excuse me? Sister? I'm the girlfriend. Don't make it weird, babe. 🙄"

    # --- GENERATION SETTINGS (For humor & speed) ---
    new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    
    chat_history_ids = model.generate(
        new_user_input_ids, 
        max_length=60,         # Super short for max speed!
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100,             # High top_k = more random/funny words
        top_p=0.85,
        temperature=1.2        # High temp = more "unhinged" and creative
    )
    
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # --- THE SASS FILTER (Fixes the "Dry" replies) ---
    slang = [" fr fr", " 💅", " lol literally", " don't play with me", " vibe check?", " ✨", " 🙄", " anyway..."]
    
    if len(response) < 5 or "i don't know" in response.lower():
        response = "You really thought you did something there, didn't you? 💅"
    else:
        # Randomly add a slang word to make it sound more "Gen Z"
        response = response + random.choice(slang)
        
    return response

# --- LUXURY PURPLE UI ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 💜 SOFI-AI")
    gr.Markdown("### *Sarcastic, iconic, and zero chill.*")
    gr.ChatInterface(fn=chat_with_sofi)

demo.launch()
