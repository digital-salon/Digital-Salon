#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragColor;

in vec4 VertPosition;
in vec4 VertNormal;
in vec4 VertColor;
in vec4 VertTexture;

uniform vec3 lightPos;

void main()
{
   vec4 color = VertColor;

   vec3 N = normalize(VertNormal.xyz);
   vec3 L = normalize(lightPos.xyz);

   float d = max(0.0, dot(N, L));

   color.xyz *= d;

   FragColor = vec4(color);	
}
