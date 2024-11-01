#version 400 core

#define FRAG_COLOR 0
#define MAX_LIGHTS 10

layout(location = FRAG_COLOR) out vec4 FragColor;

uniform struct Light 
{
    vec3 position;
    vec3 direction;
    vec3 intensity;
    vec3 attenuation;
    vec4 cone;
    int type;
    vec3 color;
} 
lights[MAX_LIGHTS];

uniform struct Material
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    float Ns;
    int hasTex;
} material;

uniform vec3 camPos;
uniform float shadowIntensity;
uniform int renderMode;
uniform float alpha;
uniform int applyShadow;
uniform int numLights;
uniform int isSelected;
uniform vec3 lightPosition;

uniform sampler2D tex; 
uniform sampler2D shadowMap[MAX_LIGHTS];

in vec4 VertPosition;
in vec4 VertColor;
in vec4 VertTexture;
in vec4 VertNormal;
in vec4 VertShadowCoord[MAX_LIGHTS];

void main()
{
    vec4 color = vec4(VertColor.x, VertColor.y, VertColor.z, 1);
    vec3 P = VertPosition.xyz;
    vec3 N = normalize(VertNormal.xyz);
    vec3 V = normalize(camPos.xyz-P);
	vec3 L = normalize(lightPosition - P);

	float ambient = 0.2;
	float diffuse = 0.9 * max(0, dot(N, L));
	vec3 texColor = vec3(0.3,0.5,0.8);//color.xyz;//texture(tex, VertTexture.xy).rgb;
	color.xyz = texColor * (ambient + diffuse);
	FragColor = color;
}
