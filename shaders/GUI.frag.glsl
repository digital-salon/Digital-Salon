#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragColor;

in vec4 VertPosition;
in vec3 VertNormal;
in vec4 VertColor;
in vec4 VertTexture;

void main()
{
   vec4 color = VertColor;
   FragColor = VertColor;	
}
