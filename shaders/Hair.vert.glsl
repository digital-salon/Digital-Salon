#version 400 core

#define VERT_POSITION	0
#define VERT_NORMAL     1
#define VERT_COLOR		2
#define VERT_TEXTURE    3

layout(location = VERT_POSITION) in vec4 Position;
layout(location = VERT_NORMAL)   in vec4 Normal;
layout(location = VERT_COLOR)    in vec4 Color;
layout(location = VERT_TEXTURE)  in vec4 Texture;

out vec4  VertPosition;
out vec4  VertNormal;
out vec4  VertColor;
out vec4  VertTexture;
out float VertThickness;
out float VertSemanticLabel;

uniform mat4 matModel;

void main()
{	      
    VertPosition = Position;
    //VertNormal   = Normal.xyz;          // Direction
	//VertColor    = Color;	            // V from PTF, VertColor.w = thick
	//VertTexture  = Texture;             // Tangent   
	
    VertNormal   = Color;          // Direction
	VertColor    = Texture;	            // V from PTF, VertColor.w = thick
	VertTexture  = Normal;             // Tangent   	
	VertThickness = Texture.w;
	VertSemanticLabel = Normal.w;
}
