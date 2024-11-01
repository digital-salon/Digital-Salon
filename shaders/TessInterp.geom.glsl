#version 400 core
#extension GL_EXT_geometry_shader4 : enable

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 3) out;
 
uniform mat4 matModel;
uniform mat4 matView;
uniform mat4 matProjection;
uniform mat4 matNormal;

in vec4 EvalPosition[];
in vec4 EvalColor[];
in vec4 EvalNormal[];
in vec4 EvalTexture[];

out vec4 GeomTexture;
out vec4 GeomColor;
out vec4 GeomNormal;
out vec4 GeomPosition;


void main()
{
    vec3 P0 = EvalPosition[0].xyz;
    vec3 P1 = EvalPosition[1].xyz;
    vec3 P2 = EvalPosition[2].xyz;

    vec3 N0 = EvalNormal[0].xyz;
    vec3 N1 = EvalNormal[1].xyz;
    vec3 N2 = EvalNormal[2].xyz;	

    vec3 wP0 = ((matModel * vec4(P0, 1)).xyz);
    vec3 wP1 = ((matModel * vec4(P1, 1)).xyz);
    vec3 wP2 = ((matModel * vec4(P2, 1)).xyz);

    vec3 wN0 = ((inverse(transpose(matModel)) * vec4(N0, 1)).xyz);
    vec3 wN1 = ((inverse(transpose(matModel)) * vec4(N1, 1)).xyz);
    vec3 wN2 = ((inverse(transpose(matModel)) * vec4(N2, 1)).xyz);	

    //Vertex 1
	GeomPosition = vec4(P0, 1);
    GeomNormal   = vec4(wN0, 1);
    GeomColor    = EvalColor[0];
    GeomTexture  = EvalTexture[0];	

    gl_Position = matProjection * matView * matModel * vec4(P0, 1);
    EmitVertex();
    
    //Vertex 2
	GeomPosition = vec4(P1, 1);
    GeomNormal   = vec4(wN1, 1);
    GeomColor    = EvalColor[1];
    GeomTexture  = EvalTexture[1];	

    gl_Position = matProjection * matView * matModel * vec4(P1, 1);
    EmitVertex();

    //Vertex 3
	GeomPosition = vec4(P2 , 1);
    GeomNormal   = vec4(wN2, 1);
    GeomColor    = EvalColor[2];
    GeomTexture  = EvalTexture[2];		

    gl_Position = matProjection * matView * matModel * vec4(P2 , 1);
    EmitVertex();

    EndPrimitive();
}