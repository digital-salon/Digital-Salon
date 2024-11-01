#version 400 core
#extension GL_EXT_geometry_shader4 : enable

#define MAX_LIGHTS      10

layout(lines, invocations = 1) in;
layout(triangle_strip, max_vertices = 18) out;

uniform mat4 matViewProjection;
uniform mat4 matRot;
uniform mat4 matModel;

uniform vec3 lightPos;
uniform vec3 camPos;
uniform int numLights;
uniform mat4x4 matLightView[MAX_LIGHTS];

in vec4 VertPosition[];
in vec4 VertColor[];
in vec4 VertNormal[];
in vec4 VertTexture[];
in float VertThickness[];
in float VertSemanticLabel[];

out vec4 texCoordA;
out vec4 texCoordB;
out float thickness;
out vec3 GeomPosition;
out vec4  color;
out vec3  normal;
out float diffuseFactor;
out vec4 VertShadowCoord[MAX_LIGHTS];
out float semanticLabel;

void main()
{    		
	float c = 1000;	
    float PI2 = 2 * 3.141592654;    
	
	float edgeLength = length(VertPosition[1].xyz - VertPosition[0].xyz);
	float nrVertices = 1;
	  
	//Reading Data
	int i = 0;
	vec4 posS = VertPosition[i];       
	vec4 posT = VertPosition[i+1];

	vec3 vS = VertColor[i].xyz;
	vec3 vT = VertColor[i+1].xyz;
	
	vec3 tS = VertTexture[i].xyz;
	vec3 tT = VertTexture[i+1].xyz;

	float thickS = VertThickness[i];
	float thickT = VertThickness[i+1];    
	
	float semanticLabelS = VertSemanticLabel[i];
	float semanticLabelT = VertSemanticLabel[i+1];
	  		  
	
	//Computing
	vec3 v11 = normalize(vS);        
	vec3 v12 = normalize(cross(vS, tS));
 
	vec3 v21 = normalize(vT);
	vec3 v22 = normalize(cross(vT, tT)); 

	float rS = max(0.00000001, thickS); 
	float rT = max(0.00000001, thickT);
	

	float dS = max(1, length(camPos - posS.xyz));
	float dT = max(1, length(camPos - posT.xyz));       

	int pS = 2 * int(c * rS / dS);
	int pT = 2 * int(c * rT / dT);

	int num = 3;
	pS = num;
	pT = num;

	int forMax = num; 
	
	//Light Pos
	vec4 lPos = normalize(vec4(-lightPos.x, -lightPos.y, -lightPos.z, 1));
	vec3 L = normalize(lPos.xyz);   

	for(int k=1; k<=num; k+=1)
	{
		int tempIS = int(k * pS/forMax);
		float angleS = (PI2 / pS) * tempIS;
							 
		int tempIT = int(k * pT/forMax);
		float angleT = (PI2 / pT) * tempIT;

		vec3 newPS = posS.xyz + (v11 * sin(angleS) + v12 * cos(angleS)) * 1.5*2.5*rS;
		vec3 newPT = posT.xyz + (v21 * sin(angleT) + v22 * cos(angleT)) * 1.5*2.5*rT; 


		//Source Vertex           
		vec3 N = normalize(newPT - newPS);             
		diffuseFactor = max(dot(N, L), 0.0);    
		normal = N;

		//if(rS < 0.001)
			//diffuseFactor = 0;
			//rS = 0.001;

		texCoordA = vec4(1.0 * tempIS / pS, 0, 0, 0); //vec4(1.0 * tempIS / pS, sTexY, 0, 0);
		//texCoordB = matLightView * vec4(newPS, 1); 

		thickness = rS;
		
		GeomPosition = newPS;
		semanticLabel = semanticLabelS;

		gl_Position = matViewProjection * matModel * vec4(newPS, 1);
		color = VertNormal[i];
		
		for(int i=0; i<numLights; i++)
		{
		   VertShadowCoord[i] = matLightView[i] * matModel * vec4(newPS, 1);
		}
		
		EmitVertex();
								

		//Target Vertex                               
		diffuseFactor = max(dot(N, L), 0.0);
		normal = N;
		
	   //if(rT < 0.001)
			//diffuseFactor = 1;
			//rT = 0.001;

	   
		float d0 = 1.0;
		float nrVertices = nrVertices;
		float dn = edgeLength/nrVertices;            
		
		texCoordA = vec4(1.0 * tempIT / pT, 1, 0, 0);//vec4(1.0 * tempIT / pT, tTexY, 0, 0);
		//texCoordB = matLightView * vec4(newPT, 1);

		thickness = rT;
		
		GeomPosition = newPT;
		semanticLabel = semanticLabelT;

		gl_Position = matViewProjection * matModel * vec4(newPT, 1);
		color = VertNormal[i];
		
		for(int i=0; i<numLights; i++)
		{
		   VertShadowCoord[i] = matLightView[i] * matModel * vec4(newPT, 1);
		}

		EmitVertex();  
	}		

	EndPrimitive();   	 		 			
}

