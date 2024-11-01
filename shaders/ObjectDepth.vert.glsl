#version 400 core

#define VERT_POSITION	0
#define VERT_NORMAL     1
#define VERT_COLOR		2
#define VERT_TEXTURE    3

uniform mat4x4 matModel;
uniform mat4x4 matView;
uniform mat4x4 matProjection;

uniform vec4 ClipPlane;

layout(location = VERT_POSITION) in vec4 Position;
layout(location = VERT_NORMAL)   in vec4 Normal;
layout(location = VERT_COLOR)    in vec4 Color;
layout(location = VERT_TEXTURE)  in vec4 Texture;

out vec4 VertPosition;
out vec4 VertNormal;
out vec4 VertColor;
out vec4 VertTexture;

out float gl_ClipDistance[1];

void main()
{	   
    VertPosition = matModel * Position; 
    VertNormal   = vec4(transpose(inverse(matModel)) * vec4(Normal.xyz, 1));    
	VertColor    = Color;
	VertTexture  = Texture;

    vec4 tmpClip = ClipPlane;
    tmpClip.y = 1.0;
    tmpClip.w = -0.1;
	
    gl_Position = matProjection * matView * matModel * vec4(Position.xyz, 1);
    gl_ClipDistance[0] = dot(matModel * Position-vec4(0, -0.25, 0 ,0), tmpClip);
}
