#version 400 core
#extension GL_EXT_geometry_shader4 : enable

layout(triangles, invocations = 1) in;
layout(points, max_vertices = 10) out;

#define VERT_POSITION	0
#define VERT_NORMAL     1
#define VERT_COLOR		2
#define VERT_TEXTURE    3

uniform mat4 matModel;
uniform mat4 matView;
uniform mat4 matProjection;

in vec4 VertPosition[];
in vec4 VertColor[];
in vec4 VertNormal[];
in vec4 VertTexture[];

out vec4 GeomPosition;
out vec4 GeomNormal;
out vec4 GeomColor;
out vec4 GeomTexture;

void main()
{    		
    for(int i=0; i<gl_VerticesIn; ++i)
    {		
		GeomPosition = matModel * vec4(VertPosition[i].xyz, 1);
		GeomNormal   = VertNormal[i];
		GeomColor    = VertColor[i];
		GeomTexture  = VertTexture[i];

        EmitVertex();
        EndPrimitive();        
    }

    EndPrimitive();
}