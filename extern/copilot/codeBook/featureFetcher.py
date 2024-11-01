import torch
#import clip
#from PIL import Image
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
#import json
#from write_html import *

from torch.nn import CosineSimilarity
cossim = CosineSimilarity(dim=1, eps=1e-6)

def dist(v1, v2):
      return cossim(v1, v2)


device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

def truncate_sentence(input_string, max_words=53):
    # Split the input string into a list of words
    words = input_string.split()

    # Check if the number of words exceeds the max_words limit
    if len(words) > max_words:
        # Chop the list to contain only the first max_words words
        truncated_words = words[:max_words]
        # Join the truncated list back into a string
        truncated_string = ' '.join(truncated_words)
        return truncated_string
    else:
        # If the sentence is already within the limit, return it as is
        return input_string
    
def get_feature(prompts):
    
    prompts = truncate_sentence(prompts)


    text_inputs = tokenizer(
    prompts, 
    padding="max_length", 
    return_tensors="pt",
    ).to(device)

    input_ids = text_inputs.input_ids.to(device)
    #print (input_ids.shape)
    
    #text_embeddings = torch.flatten(text_encoder(input_ids)['last_hidden_state'],1,-1)

    text_features = model.get_text_features(**text_inputs)
    
    #print (text_features.shape, text_embeddings.shape)

    return text_features, ""#, text_embeddings