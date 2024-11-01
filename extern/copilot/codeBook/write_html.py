# Data
images = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg",
    "image4.jpg",
    "image5.jpg",
    "image6.jpg"
]

descriptions = [
    "Description for images 1-3",
    "Description for images 4-6"
]

# HTML Content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Gallery</title>
    <style>
        .description-row {
            margin-bottom: 10px;
            text-align: center;
        }
        .image-row {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }
        .image-item {
            flex: 1;
            text-align: center;
        }
        .image-item img {
            max-width: 10%;
            height: auto;
        }
    </style>
</head>
<body>
"""

# # Loop through descriptions and create corresponding image rows
# for i in range(0, len(images), 3):
#     desc_index = i // 3
#     if desc_index < len(descriptions):
#         html_content += f"""
#             <div class="description-row">
#                 <p>{descriptions[desc_index]}</p>
#             </div>
#             <div class="image-row">
#         """
#         for j in range(3):
#             if i + j < len(images):
#                 img = images[i + j]
#                 html_content += f"""
#                     <div class="image-item">
#                         <img src="{img}" alt="{img}">
#                     </div>
#                 """
#         html_content += '</div>'

# # Closing tags
# html_content += """
# </body>
# </html>
# """

# # Write to HTML file
# with open("gallery.html", "w") as file:
#     file.write(html_content)
