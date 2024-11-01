#version 400 core

#define FRAG_COLOR 0
#define MAX_LIGHTS 10

layout(location = FRAG_COLOR) out vec4 FragData;

in float diffuseFactor;
in vec4 texCoordA; 
in vec4 texCoordB;
in float thickness;
in vec4 color;
in float visibility;
in vec3 normal;
in float vertexTreeId;
in vec3 GeomPosition;

uniform sampler2D shadowMap;
uniform sampler2D texBark; 
uniform int applyShadow;
uniform vec3 lightPos;

void main()
{
    float moment1 = gl_FragCoord.z;
    float moment2 = moment1 * moment1;

    FragData = vec4(moment1, moment2, 0.0, 1.0);
}
