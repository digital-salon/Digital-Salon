#version 400 core

layout(vertices = 3) out;

#define ID gl_InvocationID

uniform float TessLevelInner = 8.0;
uniform float TessLevelOuter = 8.0;

in vec4 VertPosition[];
in vec4 VertNormal[];
in vec4 VertColor[];
in vec4 VertTexture[];

out vec4 ContPosition[];
out vec4 ContNormal[];
out vec4 ContColor[];
out vec4 ContTexture[];

patch out vec3 b300;
patch out vec3 b030;
patch out vec3 b003;
patch out vec3 b210;
patch out vec3 b120;
patch out vec3 b021;
patch out vec3 b012;
patch out vec3 b102;
patch out vec3 b201;
patch out vec3 b111;

patch out vec3 n200;
patch out vec3 n020;
patch out vec3 n002;
patch out vec3 n110;
patch out vec3 n011;
patch out vec3 n101;

float w(vec3 Pi, vec3 Pj, vec3 Ni)
{
	return dot((Pj - Pi), Ni);
}

float v(vec3 Pi, vec3 Pj, vec3 Ni, vec3 Nj)
{
	float a = dot(Pj - Pi, Ni + Nj);
	float b = dot(Pj - Pi, Pj - Pi);
	return 2 * (a / b);
}

void main()
{
	gl_TessLevelInner[0] = TessLevelInner;
	gl_TessLevelOuter[0] = TessLevelOuter;
	gl_TessLevelOuter[1] = TessLevelOuter;
	gl_TessLevelOuter[2] = TessLevelOuter;

	ContPosition[ID] = VertPosition[ID];
	ContColor[ID]    = VertColor[ID];
	ContNormal[ID]   = VertNormal[ID];
	ContTexture[ID]  = VertTexture[ID];

	vec3 P1 = VertPosition[0].xyz;
	vec3 P2 = VertPosition[1].xyz;
	vec3 P3 = VertPosition[2].xyz;

	vec3 N1 = VertNormal[0].xyz;
	vec3 N2 = VertNormal[1].xyz;
	vec3 N3 = VertNormal[2].xyz;

	b300 = P1;
	b030 = P2;
	b003 = P3;
	b210 = (2 * P1 + P2 - w(P1, P2, N1) * N1) / 3.0f;
	b120 = (2 * P2 + P1 - w(P2, P1, N2) * N2) / 3.0f;
	b021 = (2 * P2 + P3 - w(P2, P3, N2) * N2) / 3.0f;
	b012 = (2 * P3 + P2 - w(P3, P2, N3) * N3) / 3.0f;
	b102 = (2 * P3 + P1 - w(P3, P1, N3) * N3) / 3.0f;
	b201 = (2 * P1 + P3 - w(P1, P3, N1) * N1) / 3.0f;
	vec3 E = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0f;
	vec3 V = (P1 + P2 + P3) / 3.0f;
	b111 = E + (E - V) / 2.0f;

	n200 = N1;
	n020 = N2;
	n002 = N3;
	n110 = normalize(N1 + N2 - v(P1, P2, N1, N2) * (P2 - P1));
	n011 = normalize(N2 + N3 - v(P2, P3, N2, N3) * (P3 - P2));
	n101 = normalize(N3 + N1 - v(P3, P1, N3, N1) * (P1 - P3));
   
}

