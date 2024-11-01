#version 400 core

//fractional_odd_spacing fractional_even_spacing equal_spacing
layout(triangles, equal_spacing, ccw) in;

in vec4 ContPosition[];
in vec4 ContNormal[];
in vec4 ContColor[];
in vec4 ContTexture[];

out vec4 EvalPosition;
out vec4 EvalNormal;
out vec4 EvalColor;
out vec4 EvalTexture;

patch in vec3 b300;
patch in vec3 b030;
patch in vec3 b003;
patch in vec3 b210;
patch in vec3 b120;
patch in vec3 b021;
patch in vec3 b012;
patch in vec3 b102;
patch in vec3 b201;
patch in vec3 b111;
	  
patch in vec3 n200;
patch in vec3 n020;
patch in vec3 n002;
patch in vec3 n110;
patch in vec3 n011;
patch in vec3 n101;

uniform float modifier;

vec4 interpolate(in vec4 v0, in vec4 v1, in vec4 v2)
{
	 vec3 p0 = gl_TessCoord.x * v0.xyz;
     vec3 p1 = gl_TessCoord.y * v1.xyz;
     vec3 p2 = gl_TessCoord.z * v2.xyz;

	 return vec4(p0 + p1 + p2, 1);
}

void main()
{
	float u = gl_TessCoord.x;
	float v = gl_TessCoord.y;
	float w = gl_TessCoord.z;

	float u2 = u * u;
	float v2 = v * v;
	float w2 = w * w;

	float u3 = u2 * u;
	float v3 = v2 * v;
	float w3 = w2 * w;

	vec3 pos = b300 * w3 + b030 * u3 + b003 * v3
			 + b210 * 3 * w2 * u + b120 * 3 * w * u2 + b201 * 3 * w2 * v
			 + b021 * 3 * u2 * v + b102 * 3 * w * v2 + b012 * 3 * u * v2
			 + b111 * 6 * w * u * v;

	vec3 n = n200 * w2 + n020 * u2 + n002 * v2
		   + n110 * w * u + n011 * u * v + n101 * w * v;

	n = normalize(n);

	//EvalPosition = interpolate(ContPosition[0], ContPosition[1], ContPosition[2]);
	//EvalNormal = interpolate(ContNormal[0], ContNormal[1], ContNormal[2]);
	EvalColor = interpolate(ContColor[0], ContColor[1], ContColor[2]);
	EvalTexture = interpolate(ContTexture[0], ContTexture[1], ContTexture[2]);

	pos.xyz += n * modifier;
	EvalPosition = vec4(pos,1);
	EvalNormal = vec4(n, 1);
}