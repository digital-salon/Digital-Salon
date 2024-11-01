#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragColor;

in vec4 VertPosition;
in vec4 VertNormal;
in vec4 VertColor;
in vec4 VertTexture;
in vec4 VertExtra;

uniform vec3 lightPosition;
uniform int drawWeight;

void main()
{
   vec3 P = VertPosition.xyz;
   vec3 N = normalize(VertNormal.xyz);
   vec3 L = normalize(lightPosition - P);
   
   float gv = 1.0;
   vec4 color = VertColor;//vec4(0.3, 0.5, 0.8, 1.0);
   //float d = max(0, dot(N, L));
   //color.xyz *= d;
   
   	if(drawWeight > 0.5)
	{
		vec3 red = vec3(1.0,0.0,0.0);
		vec3 blue = vec3(0.0,0.0,1.0);
		vec3 weightCol = VertExtra.x * red + (1-VertExtra.x) * blue;
		color.xyz = 0.5 * color.xyz + weightCol;
	}
	
   FragColor = color;
}
