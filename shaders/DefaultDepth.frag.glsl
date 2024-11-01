#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragData;

in vec4 VertPosition;
in vec4 VertNormal;
in vec4 VertColor;
in vec4 VertTexture;

void main()
{
    float moment1 = gl_FragCoord.z;
    float moment2 = moment1 * moment1;

    FragData = vec4(moment1, moment2, 0.0, 1.0);
}
