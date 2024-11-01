#version 400 core

#define FRAG_COLOR 0
layout(location = FRAG_COLOR) out vec4 FragColor;

in vec4 VertPosition;
in vec4 VertNormal;
in vec4 VertColor;
in vec4 VertTexture;

uniform float ScaleFactor = 0.8;

const float C1 = 0.429043;
const float C2 = 0.511664;
const float C3 = 0.743125;
const float C4 = 0.886227;
const float C5 = 0.247708;

////Old Town Square
//const vec3 L00 = vec3( 0.871297, 0.875222, 0.864470);
//const vec3 L1m1 = vec3( 0.175058, 0.245335, 0.312891);
//const vec3 L10 = vec3( 0.034675, 0.036107, 0.037362);
//const vec3 L11 = vec3(-0.004629, -0.029448, -0.048028);
//const vec3 L2m2 = vec3(-0.120535, -0.121160, -0.117507);
//const vec3 L2m1 = vec3( 0.003242, 0.003624, 0.007511);
//const vec3 L20 = vec3(-0.028667, -0.024926, -0.020998);
//const vec3 L21 = vec3(-0.077539, -0.086325, -0.091591);
//const vec3 L22 = vec3(-0.161784, -0.191783, -0.219152); 

////Grace Cathedral
//const vec3 L00 = vec3(0.79, 0.44, 0.54);
//const vec3 L1m1 = vec3(0.39, 0.35, 0.60);
//const vec3 L10 = vec3(-0.34, -0.18, -0.27);
//const vec3 L11 = vec3(-0.29, -0.06, 0.01);
//const vec3 L2m2 = vec3(-0.11, -0.05, -0.12);
//const vec3 L2m1 = vec3(-0.26, -0.22, -0.47);
//const vec3 L20 = vec3(-0.16, -0.09, -0.15);
//const vec3 L21 = vec3(0.56, 0.21, 0.14);
//const vec3 L22 = vec3(0.21, -0.5, -0.30); 

//Eucalyptus Grove
const vec3 L00  = vec3(0.38, 0.43, 0.45);
const vec3 L1m1 = vec3(0.29, 0.36, 0.41);
const vec3 L10  = vec3(0.04, 0.03, 0.01);
const vec3 L11  = vec3(-0.1, -0.1, -0.09);
const vec3 L2m2 = vec3(-0.06, -0.06, 0.04);
const vec3 L2m1 = vec3(0.01, 0.01, -0.05);
const vec3 L20  = vec3(-0.09, -0.13, -0.15);
const vec3 L21  = vec3(-0.06, -0.05, -0.04);
const vec3 L22  = vec3(0.02, 0.0, -0.05); 

////St. Peter's Basilica
//const vec3 L00 = vec3(0.36, 0.26, 0.23);
//const vec3 L1m1 = vec3(0.18, 0.14, 0.13);
//const vec3 L10 = vec3(-0.02, -0.01, 0.0);
//const vec3 L11 = vec3(0.03, 0.02, 0.0);
//const vec3 L2m2 = vec3(0.02, 0.01, 0.0);
//const vec3 L2m1 = vec3(-0.05, -0.03, -0.01);
//const vec3 L20 = vec3(-0.09, -0.08, -0.07);
//const vec3 L21 = vec3(0.01, 0.0, 0.0);
//const vec3 L22 = vec3(-0.08, -0.03, 0.0); 

////Uffizi Gallery
//const vec3 L00 = vec3(0.32, 0.31, 0.35);
//const vec3 L1m1 = vec3(0.37, 0.37, 0.43);
//const vec3 L10 = vec3(0.0, 0.0, 0.0);
//const vec3 L11 = vec3(-0.01, -0.01, -0.01);
//const vec3 L2m2 = vec3(-0.02, -0.02, -0.03);
//const vec3 L2m1 = vec3(-0.01, -0.01, -0.01);
//const vec3 L20 = vec3(-0.28, -0.28, -0.32);
//const vec3 L21 = vec3(0.00, 0.00, 0.00);
//const vec3 L22 = vec3(-0.24, -0.24, -0.28); 

////Gallileo's Tomb
//const vec3 L00 = vec3(1.04, 0.76, 0.71);
//const vec3 L1m1 = vec3(0.44, 0.34, 0.34);
//const vec3 L10 = vec3(-0.22, -0.18, -0.17);
//const vec3 L11 = vec3(0.71, 0.54, 0.56);
//const vec3 L2m2 = vec3(0.64, 0.50, 0.52);
//const vec3 L2m1 = vec3(-0.12, -0.09, -0.08);
//const vec3 L20 = vec3(-0.37, -0.28, -0.29);
//const vec3 L21 = vec3(-0.17, -0.13, -0.13);
//const vec3 L22 = vec3(0.55, 0.42, 0.42); 

////Vine Street Kitchen
//const vec3 L00 = vec3(0.64, 0.67, 0.73);
//const vec3 L1m1 = vec3(0.28, 0.32, 0.33);
//const vec3 L10 = vec3(0.42, 0.60, 0.77);
//const vec3 L11 = vec3(-0.05, -0.04, -0.02);
//const vec3 L2m2 = vec3(-0.10, -0.08, -0.05);
//const vec3 L2m1 = vec3(0.25, 0.39, 0.53);
//const vec3 L20 = vec3(0.38, 0.54, 0.71);
//const vec3 L21 = vec3(0.06, 0.01, -0.02);
//const vec3 L22 = vec3(-0.03, -0.02, -0.03); 

////Breezeway
//const vec3 L00 = vec3(0.32, 0.36, 0.38);
//const vec3 L1m1 = vec3(0.37, 0.41, 0.45);
//const vec3 L10 = vec3(-0.01, -0.01, -0.01);
//const vec3 L11 = vec3(-0.10, -0.12, -0.12);
//const vec3 L2m2 = vec3(-0.13, -0.15, -0.17);
//const vec3 L2m1 = vec3(-0.01, -0.02, 0.02);
//const vec3 L20 = vec3(-0.07, -0.08, -0.09);
//const vec3 L21 = vec3(0.02, 0.03, 0.003);
//const vec3 L22 = vec3(-0.29, -0.32, -0.36); 

////Campus Sunset 
//const vec3 L00  = vec3(0.79, 0.94, 0.98);
//const vec3 L1m1 = vec3(0.44, 0.56, 0.70);
//const vec3 L10  = vec3(-0.10, -0.18, -0.27);
//const vec3 L11  = vec3(0.45, 0.38, 0.20);
//const vec3 L2m2 = vec3(0.18, 0.14, 0.05);
//const vec3 L2m1 = vec3(-0.14, -0.22, -0.31);
//const vec3 L20  = vec3(-0.39, -0.40, -0.36);
//const vec3 L21  = vec3(0.09, 0.07, 0.04);
//const vec3 L22  = vec3(0.67, 0.67, 0.52); 

////Funston Beach Sunset
//const vec3 L00 = vec3(0.68, 0.69, 0.70);
//const vec3 L1m1 = vec3(0.32, 0.37, 0.44);
//const vec3 L10 = vec3(-0.17, -0.17, -0.17);
//const vec3 L11 = vec3(-0.45, -0.42, -0.34);
//const vec3 L2m2 = vec3(-0.17, -0.17, -0.15);
//const vec3 L2m1 = vec3(-0.08, -0.09, -0.10);
//const vec3 L20 = vec3(-0.03, -0.02, -0.01);
//const vec3 L21 = vec3(0.16, 0.14, 0.10);
//const vec3 L22 = vec3(0.37, 0.31, 0.20); 

vec3 sphericalHarmonic()
{
    vec3 tnorm = VertNormal.xyz;

    vec3 diffuse = C1 * L22 * (tnorm.x * tnorm.x - tnorm.y * tnorm.y) + 
    C3 * L20 * tnorm.z * tnorm.z +
    C4 * L00 -
    C5 * L20 +
    2.0 * C1 * L2m2 * tnorm.x * tnorm.y +
    2.0 * C1 * L21 * tnorm.x * tnorm.z +
    2.0 * C1 * L2m1 * tnorm.y * tnorm.z +
    2.0 * C2 * L11 * tnorm.x +
    2.0 * C2 * L1m1 * tnorm.y +
    2.0 * C2 * L10 * tnorm.z;

    diffuse *= ScaleFactor; 

    return diffuse;
}

void main()
{
   vec4 color = vec4(sphericalHarmonic(), 1);
   FragColor = color;	
}
