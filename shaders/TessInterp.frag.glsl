#version 400 core

#define FRAG_COLOR 0

layout(location = FRAG_COLOR) out vec4 FragColor;

in vec4 GeomTexture;
in vec4 GeomColor;
in vec4 GeomNormal;
in vec4 GeomPosition;

uniform vec3 lightPos;
uniform vec3 camPos;

void main()
{
    vec3 P = GeomPosition.xyz;
    vec3 N = normalize(GeomNormal.xyz);
    vec3 L = normalize(lightPos.xyz - P);
    vec3 V = normalize(camPos.xyz-P);
    vec3 H = normalize(L+V);

    float d = max(dot(N, L), 0);
    float s = pow(max(dot(N, H), 0), 20);

    if(d <= 0)
        s = 0;
	
    vec3 lColor      = vec3(1, 1, 1);
    vec3 matAmbient  = vec3(0.0, 0.0, 0.0);
    vec3 matDiffuse  = vec3(1, 1, 1);
    vec3 matSpecular = vec3(1, 1, 1);

    vec3 ambient  = matAmbient  * lColor;
    vec3 diffuse  = matDiffuse  * lColor * d;
    vec3 specular = matSpecular * lColor * s;
    
    vec4 color = vec4(ambient + diffuse + specular, 1);

    FragColor = color;
}
