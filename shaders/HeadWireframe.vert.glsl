#version 400 core

#define VERT_POSITION	0
#define VERT_NORMAL     1
#define VERT_COLOR		2
#define VERT_TEXTURE    3
#define VERT_EXTRA 		4

uniform mat4x4 matModel;
uniform mat4x4 matView;
uniform mat4x4 matProjection;

layout(location = VERT_POSITION) in vec4 Position;
layout(location = VERT_NORMAL)   in vec4 Normal;
layout(location = VERT_COLOR)    in vec4 Color;
layout(location = VERT_TEXTURE)  in vec4 Texture;
layout(location = VERT_EXTRA)    in vec4 Extra;

out vec4 VertPosition;
out vec4 VertNormal;
out vec4 VertColor;
out vec4 VertTexture;
out vec4 VertExtra;

void main()
{	   
    VertPosition = Position; 
    VertNormal   = Normal;
	VertColor    = Color;
	VertTexture  = Texture;
	VertExtra    = Extra;
	
    gl_Position = matProjection * matView * matModel * vec4(Position.xyz, 1);
}
