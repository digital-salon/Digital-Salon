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

uniform vec3 ambient;
uniform vec3 camPos;
uniform float shadowIntensity;
uniform int renderMode;
uniform float alpha;
uniform int applyShadow;
uniform int numLights;

uniform sampler2D tex; 
uniform sampler2D shadowMap[MAX_LIGHTS];

in vec4 VertPosition;
in vec4 VertColor;
in vec4 VertTexture;
in vec4 VertNormal;
in vec4 VertShadowCoord[MAX_LIGHTS];

float lightAttenuation(vec3 P, vec3 L, vec3 attenuation, float intensity)
{
    float d = distance(P, L);
    float r = 1 / (attenuation.x + attenuation.y* d + attenuation.z * d * d) * intensity;
    return r;
}

float dualConeSpotlight(vec3 P, vec3 LP, vec3 LD, float outerCone, float innerCone)
{
    vec3 V = normalize(P - LP);
    float d = dot(V, LD);
    return smoothstep(outerCone, innerCone, d);
}

float VSM(vec4 smcoord, sampler2D sm)
{
	vec3 coords = smcoord.xyz / smcoord.w ;    

    if(smcoord.z < 1)
        return 1.0;

    float depth = 1.0;//coords.z;
    
    vec4 depthBlurrred = texture(sm, coords.xy);

    float depthSM = depthBlurrred.x;
	float sigma2  = depthBlurrred.y;

    float realDepth = texture(sm, coords.xy).x;

	sigma2 -= depthSM * depthSM;

	float bias = 0.0001;

	float dist = depth - depthSM;
	float P = sigma2 / ( sigma2 + dist * dist );
	float lit = max( P, ( depth - bias ) <= depthSM ? 1.0 : 0.0);
	lit = min(1.0, lit);   

    return mix(shadowIntensity, 1.0, lit);

    return lit;
}

vec3 applyLight(Light light, Material material, vec3 P, vec3 N, vec3 V)
{
    vec3 texColor = vec3(1, 1, 1);    
    vec3 diffuse = vec3(0);
    vec3 specular = vec3(0);

    vec3 L = normalize(light.position - P);
       

    if(material.hasTex == 1)
    {
        texColor = texture(tex, VertTexture.st).rgb;
    }

	vec3 ambient =  material.Ka * light.color * texColor;    


    float d = max(dot(N, L), 0);        
    float s = 0;

    if(d >= 0)
    {
        vec3 H = normalize(L+V);
        s = pow(max(dot(N, H), 0), material.Ns);
        specular = material.Ks * light.color * s * 1.0;
    }
	
    diffuse  = material.Kd * texColor.rgb * light.color * d;    

    float t = dualConeSpotlight(P, light.position, -light.direction, 21.0, 32.0);
    float a = lightAttenuation(P, light.position.xyz-P, light.attenuation, light.intensity.x);

    vec3 color = ambient + a * (diffuse + specular);

    return color;
}

void main()
{
    vec4 color = vec4(0, 0, 0, 1);

    float shadow = 1.0;

    vec3 P = VertPosition.xyz;
    vec3 N = normalize(VertNormal.xyz);
    vec3 V = normalize(camPos.xyz-P);
   
    if(applyShadow == 1)    
    {
        for(int i=0; i<numLights; i++)
        {            
            shadow *= VSM(VertShadowCoord[i], shadowMap[i]); 
        }   
    }

    if(renderMode == 0)
    {
        for(int i=0; i<numLights; i++)
        {
            color.rgb += applyLight(lights[i], material, P, N, V) * shadow;
        }   
        color.a = alpha;
    }
    if(renderMode == 1)
    {
        color.xyz = vec3(0.2, 0.2, 0.2) * shadow;
        color.a = alpha;
    }
    if(renderMode == 2)
    {
        float d = 1;
        for(int i=0; i<numLights; i++)
        {
            vec3 L = normalize(lights[i].position.xyz - P);        
            d *= max(0, dot(N, L));
        }

        color.xyz = vec3(1.0, 1.0, 1.0) * d * shadow;
        color.a = alpha;
    }
    if(renderMode == 3)
    {
        color.xyz = vec3(1.0, 1.0, 1.0) * shadow;
        color.a = alpha;
    }
    if(renderMode == 4)
    {
        discard;
    }

    color.rgb += ambient;
    FragColor = vec4(color.rgb, alpha);	
}