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

uniform vec3 camPos;
uniform float shadowIntensity;
uniform int renderMode;
uniform float alpha;
uniform int applyShadow;
uniform int numLights;
uniform int isSelected;

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
        return 1;

    float depth = coords.z;
    
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
    vec3 ambient =  material.Ka * light.color;
    vec3 diffuse = vec3(0);
    vec3 specular = vec3(0);

    vec3 L = normalize(light.position - P);
       
    if(material.hasTex == 1)
    {
        texColor = texture(tex, VertTexture.st).rgb;
    }    

    float d = max(dot(N, L), 0);        
    float s = 0;

    if(d >= 0)
    {
        vec3 H = normalize(L+V);
        s = pow(max(dot(N, H), 0), material.Ns);
        specular = material.Ks * light.color * s * 1;
    }

    vec3 selecColor = vec3(1);
    if(isSelected == 1)
    {
        selecColor = vec3(1, 0, 0);
    }
	
    diffuse  = material.Kd * selecColor * texColor.rgb * light.color * d;    

    float t = dualConeSpotlight(P, light.position, -light.direction, 21.0, 32.0);
    float a = lightAttenuation(P, light.position.xyz-P, light.attenuation, light.intensity.x);

    vec3 color = ambient + (diffuse + specular) * a;

    return color;
}

vec3 cookTorrance(vec3 lightPos)
{
    // set important material values
    float roughnessValue = 0.3; // 0 : smooth, 1: rough
    float F0 = 0.8; // fresnel reflectance at normal incidence
    float k = 0.2; // fraction of diffuse reflection (specular reflection = 1 - k)
    //vec3 lightColor = vec3(1);
    vec3 lightColor = vec3(1);

    
    // interpolating normals will change the length of the normal, so renormalize the normal.
    vec3 normal = normalize(VertNormal.xyz);
    vec3 lightDirection = normalize(lightPos - VertPosition.xyz);
    vec3 varEyeDir = normalize(camPos - VertPosition.xyz);
    
    // do the lighting calculation for each fragment.
    float NdotL = max(dot(normal, lightDirection), 0.0);
    
    float specular = 0.0;
    if(NdotL > 0.0)
    {
        vec3 eyeDir = normalize(varEyeDir);

        // calculate intermediary values
        vec3 halfVector = normalize(lightDirection + eyeDir);
        float NdotH = max(dot(normal, halfVector), 0.0); 
        float NdotV = max(dot(normal, eyeDir), 0.0); // note: this could also be NdotL, which is the same value
        float VdotH = max(dot(eyeDir, halfVector), 0.0);
        float mSquared = roughnessValue * roughnessValue;
        
        // geometric attenuation
        float NH2 = 2.0 * NdotH;
        float g1 = (NH2 * NdotV) / VdotH;
        float g2 = (NH2 * NdotL) / VdotH;
        float geoAtt = min(1.0, min(g1, g2));
     
        // roughness (or: microfacet distribution function)
        // beckmann distribution function
        float r1 = 1.0 / ( 4.0 * mSquared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (mSquared * NdotH * NdotH);
        float roughness = r1 * exp(r2);
        
        // fresnel
        // Schlick approximation
        float fresnel = pow(1.0 - VdotH, 5.0);
        fresnel *= (1.0 - F0);
        fresnel += F0;
        
        specular = (fresnel * geoAtt * roughness) / (NdotV * NdotL * 3.14592654);
    }
    
    vec3 finalValue = lightColor * NdotL * (k + specular * (1.0 - k));

    return vec3(finalValue);
}


#define ATTENUATE_THRESH 0.5
#define ROUGHMAP 0.5
vec3 LightContribution(in vec3 lightPos, in vec3 N, in vec3 V)
{
    float NdotV = dot(N, V);
    vec3 L = -normalize(VertPosition.xyz - lightPos.rgb);
    float lightW = 1.0;

    vec3 lightAmbient = vec3(0.1);
    vec3 lightDiffuse = vec3(1);
    vec3 lightSpecular = vec3(0.5);

    //vec4 diffuse = vec4(0.4, 0.4, 1, 1);
    vec4 diffuse = vec4(0.8, 0.0, 0, 1);
    vec4 specular = vec4(1.0);
    vec4 roughness = vec4(1.4);
    float R_2 = 10.0;

    float constantAttenuation = 0.01;
    float linearAttenuation = 0.01;
    float quadraticAttenuation = 0.002;

	/* calculate light attenuation due to distance (do this first so we can return early if possible) */
	float attenuate = 1.0;

	if (bool(lightW)) /* directional sources don't get attenuated */
    {
		float dist = length((lightPos).xyz - VertPosition.xyz);
		float attenDiv = (constantAttenuation + linearAttenuation * dist + quadraticAttenuation * dist * dist);
		// if none of the attenuation parameters are set, we keep 1.0
		if (bool(attenDiv)) 
        {
			attenuate = 1.0 / attenDiv;
		}
	}

	/* if we're out of range, ignore the light; else calculate its contribution */
	if(attenuate < ATTENUATE_THRESH) {
		return vec3(0.0);
	}

	/* Normalize vectors and cache dot products */
	float NdotL = clamp(dot(N, L), 0.0, 1.0);

	/* Compute the final color contribution of the light */
	vec3 ambientColor = diffuse.rgb * diffuse.a * lightAmbient.rgb;
	vec3 diffuseColor = diffuse.rgb * diffuse.a * lightDiffuse.rgb * NdotL;
	vec3 specularColor;

	/* Cook-Torrance shading */
	if (ROUGHMAP > 0) 
    {
		vec3 H = normalize(L + V);
		float NdotH = dot(N, H);
		float VdotH = dot(V, H);
		float NdotH_2 = NdotH * NdotH;

		/* Compute the geometric term for specularity */
		float G1 = (2.0 * NdotH * NdotV) / VdotH;
		float G2 = (2.0 * NdotH * NdotL) / VdotH;
		float G = clamp(min(G1, G2), 0.0, 1.0);

		/* Compute the roughness term for specularity */
		float A = 1.0 / (4.0 * R_2 * NdotH_2 * NdotH_2);
		float B = exp((NdotH_2 - 1.0) / (R_2 * NdotH_2));
		float R = A * B;

		/* Compute the fresnel term for specularity using Schlick's approximation*/
		float F = roughness.g + (1.0 - roughness.g) * pow(1.0 - VdotH, 5.0);

		specularColor = lightSpecular.rgb * specular.rgb * roughness.b * NdotL * (F * R * G) / (NdotV * NdotL);
	} 
    else 
    { /* Phong shading */
		specularColor = lightSpecular.rgb * specular.rgb * pow(max(dot(V, reflect(L, N)), 0.0), specular.a);
	}

	/* @note We attenuate light here, but attenuation doesn't affect "directional" sources like the sun */
	return (attenuate * (max(ambientColor, 0.0) + max(diffuseColor, 0.0) + max(specularColor, 0.0)));
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

    for(int i=0; i<numLights; i++)
    {                
        color.rgb += LightContribution(lights[i].position, N, V);     
        //color.rgb += cookTorrance(lights[i].position);     

    }   
    color.a = alpha;

    FragColor = vec4(color.xyz, 1);	
}
