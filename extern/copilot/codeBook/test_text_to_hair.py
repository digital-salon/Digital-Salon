import torch
from PIL import Image
import json
from write_html import *

from featureFetcher import get_feature, dist,device


data_root="/home/papagina/Documents/proj_hair/digital_salon/s3/"

label_folder = data_root+"label/"

features_fn = label_folder+"features.json"

test_fn = label_folder+"test_prompts.json"


with open(features_fn, 'r') as f:
    code_book = json.load(f)

## prepare the code matrix

code_len = len(code_book[0]["title_code"][0])
code_num = len(code_book)

title_matrix = torch.ones(code_num, code_len).to(device)
description_matrix = torch.ones(code_num, code_len).to(device)

for i in range(code_num):
    c= torch.FloatTensor(code_book[i]["title_code"]).to(device).view(-1)
    title_matrix[i]=c

    c= torch.FloatTensor(code_book[i]["description_code"]).to(device).view(-1)
    description_matrix[i]=c

    
print ("finish loading the code matrix.t")

with open(test_fn, 'r') as f:
    test_prompts = json.load(f)

for test_prompt in test_prompts:
    
    print ("####")
    print (test_prompt["description"])

    # write into html
    html_content += f"""
            <div class="description-row">
                <p>{test_prompt["description"]}</p>
            </div>
        """
    
    text_feature, _ = get_feature(test_prompt["description"])
    
    print (text_feature.shape, title_matrix.shape)

    if(len(test_prompt["description"].split())<11):

        # Step 2: Find the top three similarities for titles
        similarities = dist(title_matrix, text_feature).view(-1)  # Resulting in a tensor of size N
        print (similarities.shape)
        topk_values, topk_indices = torch.topk(similarities, 3, largest=True)
        print (topk_values)
        print (topk_indices)
        
        html_content += f"""
            <div class="image-row">
        """
        for i in topk_indices:
            
            #img_fn = "extra_complex/"+code_book[i]['name'][0:-4]
            
            img_fn = code_book[i]["hair_path"]

            #print (img_fn)
            img_fn = data_root+"image/"+img_fn+"_cam.001.jpg"
            html_content += f"""
            <img src="{img_fn}" alt="img_fn" width=10%>
            """
            
        html_content += '   </div>\n'

    else:

        # Step 3: Find the top three similarities for descriptions
        similarities = dist(description_matrix, text_feature).view(-1)  # Resulting in a tensor of size N
        topk_values, topk_indices = torch.topk(similarities, 3, largest=True)
        print (topk_values)
        print (topk_indices)
        
        html_content += f"""
            <div class="image-row">
        """
        for i in topk_indices:
            
            #img_fn = "extra_complex/"+code_book[i]['name'][0:-4]
            
            img_fn = code_book[i]["hair_path"]

            #print (img_fn)
            img_fn = data_root+"image/"+img_fn+"_cam.001.jpg"
            html_content += f"""
            <img src="{img_fn}" alt="img_fn" width=10%>
            """
            
        html_content += '   </div>\n'

    


# Write to HTML file
with open("gallery.html", "w") as file:
    file.write(html_content)
