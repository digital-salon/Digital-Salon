#version 400 core

#define FRAG_COLOR 0
#define MAX_LIGHTS 10

#define FRAG_DATA 0
layout(location = FRAG_DATA) out vec4 FragColor;


in vec4 color;
in vec3 normal;
in vec3 GeomPosition;
in vec4 VertShadowCoord[MAX_LIGHTS];

uniform float shadowIntensity;
uniform sampler2D shadowMap[MAX_LIGHTS];  
uniform int applyShadow;
uniform vec3 lightPos;
uniform vec3 camPos;
uniform int numLights;
uniform float lightIntensity;
uniform vec3 strandColor;
uniform int hairDebbug;

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

float lightAttenuation(vec3 P, vec3 L, vec3 attenuation, float intensity)
{
    float d = distance(P, L);
    float r = 1 / (attenuation.x + attenuation.y* d + attenuation.z * d * d) * intensity;
    return r;
}

float lookup(sampler2D sm, vec2 offSet, vec4 sCoord)
{
    float unitX = 1.0/(1069);
    float unitY = 1.0/(1096);

    vec4 coords = sCoord / sCoord.w;

    float x = (offSet.x * unitX);
    float y = (offSet.y * unitY);

    float expDepth = texture2D(sm, coords.xy + vec2(x, y)).x;

    return expDepth;
}

float calcShadow(vec4 ShadowCoord, sampler2D sm)
{
    float shadow = 0.0;	
    float shadowStrenght = shadowIntensity;

    float c = 0.0;
    float r = 1.0;
    float s = 0.5;
    
    vec4 coord = ShadowCoord / ShadowCoord.w;

	for (float y = -r ; y <=r ; y+=s)
	{
		for (float x = -r ; x <=r ; x+=s)
		{
			float temp = lookup(sm, vec2(x,y), ShadowCoord);
			
			if(temp < (coord.z - 0.0008))
				shadow += shadowStrenght;											
			else
			    shadow += 1;

			c+=1.0;				
		}
	}
	shadow /= c ;		

    return shadow;
}

float VSM(vec4 smcoord, sampler2D sm)
{
	vec3 coords = smcoord.xyz / smcoord.w ;    

    if(smcoord.z < 1)
        return 1;

    float depth = coords.z;
    
    vec4 depthBlurrred = texture(sm, coords.xy);

    float depthSM = depthBlurrred.x;
	float sigma2  = depthBlurrred.y;

    float realDepth = texture(sm, coords.xy).x;

	sigma2 -= depthSM * depthSM;

	float bias = 0.000001; 

	float dist = depth - depthSM;
	float P = sigma2 / ( sigma2 + dist * dist );
	float lit = max( P, ( depth - bias ) <= depthSM ? 1.0 : 0.0);
	lit = min(1.0, lit);   

    return mix(shadowIntensity, 1.0, lit);

    return lit;
}


void main()
{

	// Scene vectors
	Light light = lights[0];
	vec3 T = normalize(normal);
	vec3 L = normalize(light.position - GeomPosition.xyz);
	vec3 V = normalize(camPos - GeomPosition.xyz); 
	
	float TdotL = dot(T,L);
	float TdotV = dot(T,V);
	
	vec3 L2 = normalize(vec3(0,10,0) - GeomPosition.xyz);
	float TdotL2 = dot(T,L2);
	
	// ambient
	vec3 rojito = vec3(0.82, 0.42, 0.24);
	vec3 yellow = vec3(1.0, 0.8, 0.4);
	vec3 hairColor = hairDebbug==1 ? 1.5f*vec3(color) : 2.1*strandColor; 
	vec3 lightColor = vec3(1.0, 1.0, 1.0);//vec3(0.52, 0.22, 0.84);
	vec3 ambient = lightColor * hairColor;//vec3(200.0/255.0, 100.0/255.0, 0.0/255.0);
	
	// diffuse
	float kajiyaDiff = max(0,sin(acos(TdotL)));
	float kajiyaDiff2 = max(0,sin(acos(TdotL2)));
	kajiyaDiff = pow(kajiyaDiff, 10) + 1.3*pow(kajiyaDiff2, 10);
	
	// specular 
	float kajiyaSpec = max(0,cos(abs(acos(TdotL) - acos(-TdotV))));
	float kajiyaSpec2 = max(0,cos(abs(acos(TdotL2) - acos(-TdotV))));
	kajiyaSpec = pow(kajiyaSpec, 100) + pow(kajiyaSpec2, 150);
	vec3 specColour = vec3(1.0);

	// shadow
	float shadow = 1.0; 
	if(hairDebbug==0)
    {
		shadow = 1.f;
        for(int i=0; i<numLights; i++)
        {            
            //shadow *= VSM(VertShadowCoord[i], shadowMap[i]); 
			shadow *= 0.2+calcShadow(VertShadowCoord[i], shadowMap[i]);
			//shadow *= shadow * shadow;
			//shadow = 1;
        }   
    }
	
	// Final color
	float total = 0.6;
	float amb = 0.4;
	float diff = 0.3;
	float spec = 0.1;
	vec4 colorF = vec4(1.0);
	colorF.xyz = (amb + diff * kajiyaDiff * shadow) * ambient + spec * kajiyaSpec * specColour * shadow;
	colorF.xyz *= total;
	
	// cut hair
	FragColor = vec4(colorF.xyz, 1.f);
}
