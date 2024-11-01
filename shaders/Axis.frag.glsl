#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragColor;

in vec4 VertPosition;
in vec4 VertNormal;
in vec4 VertColor;
in vec4 VertTexture;

uniform vec4 col = vec4(1.0);

void main()
{
   vec4 color = col;
   FragColor = vec4(color);	
}
