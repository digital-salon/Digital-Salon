import torch
from PIL import Image
import json
from write_html import *

from featureFetcher import get_feature, dist

    

prompt_gt="quiff, voluminous, short sides,pompadour"

prompts=["The hairstyle in the image features a prominent, voluminous quiff at the front, which rises sharply upward before tapering off towards the back. The sides are cut short, blending seamlessly into the longer top section. The overall look is clean and modern, with the quiff adding a bold, edgy element to the style."]
prompts+=["This hairstyle features two high pigtails, one on each side of the head. The hair is parted neatly down the middle, with each section gathered and secured, creating full, voluminous pigtails. The hair in the pigtails is slightly wavy, adding texture and bounce to the style. The look is playful and youthful, with a soft, rounded shape."]




# gt_text = clip.tokenize(prompt_gt).to(device)
# print ("gt_text", gt_text.shape)
# gt_features= model.encode_text(gt_text)
# print (gt_features.shape)

# test_texts = clip.tokenize(prompts).to(device)
# print ("test_texts", test_texts.shape)
# test_features= model.encode_text(test_texts)
# print (test_features.shape)

gt_features,gt_embeddings=get_feature(prompt_gt)

test_features,test_embeddings= get_feature(prompts)


for i in range(len(prompts)):
    print( prompt_gt,"->", prompts[i], "=", dist(gt_features[0], test_features[i]))

    print( prompt_gt,"->", prompts[i], "=", dist(gt_embeddings[0], test_embeddings[i]))


