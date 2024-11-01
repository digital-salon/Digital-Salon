#version 400 core

#define VERT_POSITION	0
#define VERT_NORMAL     1
#define VERT_COLOR		2
#define VERT_TEXTURE    3

#define MAX_LIGHTS      10

uniform mat4x4 matModelViewProjection;
uniform mat4x4 matLightView[MAX_LIGHTS];
uniform int numLights;

layout(location = VERT_POSITION) in vec4 Position;
layout(location = VERT_NORMAL)   in vec4 Normal;
layout(location = VERT_COLOR)    in vec4 Color;
layout(location = VERT_TEXTURE)  in vec4 Texture;

out vec4 VertPosition;
out vec4 VertNormal;
out vec4 VertColor;
out vec4 VertTexture;
out vec4 VertShadowCoord[MAX_LIGHTS];

void main()
{	   
    VertPosition = Position; 
    VertNormal   = Normal;
	VertColor    = Color;
	VertTexture  = Texture;
    
    for(int i=0; i<numLights; i++)
    {
       VertShadowCoord[i] = matLightView[i] * vec4(Position.xyz, 1);
    }

    gl_Position = matModelViewProjection * vec4(Position.xyz, 1);
}
