import torch
#import clip
#from PIL import Image
import json
from featureFetcher import get_feature

data_root="/home/papagina/Documents/proj_hair/digital_salon/s3/"

label_folder = data_root+"label/"

labels_fn = label_folder+"labels.json"

features_fn = label_folder+"features.json"
#embeddings_fn = label_folder+"embeddings.json"
features=[]
#embeddings=[]



test_fn = label_folder+"test_prompts.json"
print (labels_fn)

n=0
##### COMPUTE the text features for all labels #####
with open(labels_fn, 'r') as f:
    labels = json.load(f)

for i in range(len(labels)):
    label = labels[i]
    
    desription = label["description"]
    desription_features, desription_embeddings = get_feature(desription) 

    title = label["title"]
    title_features, title_embeddings = get_feature(title) 


    features+=[{"hair_path":label["hair_path"], "description":label["description"],"title_code":title_features.data.tolist(), "description_code":desription_features.data.tolist()}]

    #embeddings+=[{"hair_path":label["hair_path"], "title_code":title_embeddings.data.tolist(), "description_code":desription_embeddings.data.tolist()}]

    n=n+1
    
    print (n)
    


with open(features_fn, 'w') as f:
    json.dump(features, f)

# with open(embeddings_fn, 'w') as f:
#     json.dump(embeddings, f)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]