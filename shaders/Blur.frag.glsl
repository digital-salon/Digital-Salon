#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragColor;

uniform sampler2D tex; 
uniform int horizontal;
uniform vec2 params;

in vec4 VertColor;
in vec4 VertTexture;

// Relative filter weights indexed by distance from "home" texel
//    This set for 9-texel sampling
#define WT9_0 1.0
#define WT9_1 0.8
#define WT9_2 0.6
#define WT9_3 0.4
#define WT9_4 0.2

// Alt pattern -- try your own!
// #define WT9_0 0.1
// #define WT9_1 0.2
// #define WT9_2 3.0
// #define WT9_3 1.0
// #define WT9_4 0.4

#define WT9_NORMALIZE (WT9_0+2.0*(WT9_1+WT9_2+WT9_3+WT9_4))

vec4 blurH(sampler2D tex, vec2 coord, vec2 dimensions, float stride)
{
	float TexelIncrement = stride/dimensions.x;
    //float3 Coord = float3(TexCoord.xy+QuadTexelOffsets.xy, 1);

    vec2 c0 = vec2(coord.x + TexelIncrement,     coord.y);
    vec2 c1 = vec2(coord.x + TexelIncrement * 2, coord.y);
    vec2 c2 = vec2(coord.x + TexelIncrement * 3, coord.y);
    vec2 c3 = vec2(coord.x + TexelIncrement * 4, coord.y);
    vec2 c4 = vec2(coord.x,                      coord.y);
    vec2 c5 = vec2(coord.x - TexelIncrement,     coord.y);
    vec2 c6 = vec2(coord.x - TexelIncrement * 2, coord.y);
    vec2 c7 = vec2(coord.x - TexelIncrement * 3, coord.y);
    vec2 c8 = vec2(coord.x - TexelIncrement * 4, coord.y);

    vec4 OutCol;

    OutCol  = texture(tex, c0) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c1) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c2) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c3) * (WT9_4/WT9_NORMALIZE);
    OutCol += texture(tex, c4) * (WT9_0/WT9_NORMALIZE);
    OutCol += texture(tex, c5) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c6) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c7) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c8) * (WT9_4/WT9_NORMALIZE);

    return OutCol;
}

vec4 blurV(sampler2D tex, vec2 coord, vec2 dimensions, float stride)
{
	float TexelIncrement = stride/dimensions.y;
    //float3 Coord = float3(TexCoord.xy+QuadTexelOffsets.xy, 1);

    vec2 c0 = vec2(coord.x, coord.y + TexelIncrement);
    vec2 c1 = vec2(coord.x, coord.y + TexelIncrement * 2);
    vec2 c2 = vec2(coord.x, coord.y + TexelIncrement * 3);
    vec2 c3 = vec2(coord.x, coord.y + TexelIncrement * 4);
    vec2 c4 = vec2(coord.x, coord.y);
    vec2 c5 = vec2(coord.x, coord.y - TexelIncrement);
    vec2 c6 = vec2(coord.x, coord.y - TexelIncrement * 2);
    vec2 c7 = vec2(coord.x, coord.y - TexelIncrement * 3);
    vec2 c8 = vec2(coord.x, coord.y - TexelIncrement * 4);

    vec4 OutCol;

    OutCol  = texture(tex, c0) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c1) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c2) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c3) * (WT9_4/WT9_NORMALIZE);
    OutCol += texture(tex, c4) * (WT9_0/WT9_NORMALIZE);
    OutCol += texture(tex, c5) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c6) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c7) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c8) * (WT9_4/WT9_NORMALIZE);

    return OutCol;
}

vec4 blurVNew(sampler2D tex, vec2 coord, vec2 dimensions, float stride, const float sigma)
{
	float unit = stride/dimensions.y;
	int radius = min(20, max(1, 3 * int(sigma)));
    float sigmasquare = sigma*sigma;
	float gauss[20];
	
    for (int i = 0; i < radius; ++i)
    {
        gauss[i]  = (1.0f /(sqrt(2.0*3.141592*sigmasquare))) * exp(-(i*i) / (2.0*sigmasquare));	
    } 	
	
	float sum = 0;
	for (int i = 1; i < radius; i++) 
	{
		sum += abs(gauss[i]);
	}
	
	sum = 2*sum + abs(gauss[0]);
	for (int i = 0; i < radius; i++) 
	{
		gauss[i] /= sum;
	} 	
		
	vec3 color = vec3(0);
	for(int i = 1; i<radius; ++i)
	{
		vec2 c = vec2(coord.x, coord.y + unit*i);
	    color += texture(tex, c).xyz * gauss[i];
		
		c = vec2(coord.x, coord.y + unit*-i);
	    color += texture(tex, c).xyz * gauss[i];		
	}
	
	vec2 c = vec2(coord.x, coord.y);
	color += texture(tex, c).xyz * gauss[0];	

    return vec4(color.xyz, 1);
}

vec4 blurHNew(sampler2D tex, vec2 coord, vec2 dimensions, float stride, const float sigma)
{
	float unit = stride/dimensions.x;
	int radius = min(20, max(1, 3 * int(sigma)));
    float sigmasquare = sigma*sigma;
	float gauss[20];
	
    for (int i = 0; i < radius; ++i)
    {
        gauss[i]  = (1.0f /(sqrt(2.0*3.141592*sigmasquare))) * exp(-(i*i) / (2.0*sigmasquare));	
    } 	
	
	float sum = 0;
	for (int i = 1; i < radius; i++) 
	{
		sum += abs(gauss[i]);
	}
	
	sum = 2*sum + abs(gauss[0]);
	for (int i = 0; i < radius; i++) 
	{
		gauss[i] /= sum;
	} 	
		
	vec3 color = vec3(0);
	for(int i = 1; i<radius; ++i)
	{
		vec2 c = vec2(coord.x +  unit*i, coord.y);
	    color += texture(tex, c).xyz * gauss[i];
		
		c = vec2(coord.x + unit*-i, coord.y);
	    color += texture(tex, c).xyz * gauss[i];		
	}
	
	vec2 c = vec2(coord.x, coord.y);
	color += texture(tex, c).xyz * gauss[0];	

    return vec4(color.xyz, 1); 
}

void main()
{
   vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

   vec2 dim = textureSize(tex, 0);

   float sigma = params.x;
   float stride = params.y;
   
    if(horizontal == 1)
       //color = blurH(tex, VertTexture.st, dim, 2.0);
	   color = blurHNew(tex, VertTexture.st, dim, stride, sigma);
    else
       //color = blurV(tex, VertTexture.st, dim, 2.0);
	   color = blurVNew(tex, VertTexture.st, dim, stride, sigma);

   FragColor = color;	
}
