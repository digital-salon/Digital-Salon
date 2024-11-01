#include "Common.h"
#include "Geometry.h"
#include "Camera.h"
#include <assert.h>


Triangle::Triangle()
: a(),
  b(), 
  c(),
  n(), 
  fa(-1),
  fb(-1),
  fc(-1), 
  center(), 
  perVertexNormal(false), 
  selected(0)
{
}

Triangle::Triangle(const vec3& va, const vec3& vb, const vec3& vc)
	: a(va),
	b(vb),
	c(vc),
	n(),
	fa(-1),
	fb(-1),
	fc(-1),
	center(),
	perVertexNormal(false),
	selected(0)
{
	center = (a + b + c) / 3.0f;
}

Triangle::Triangle(const vec3& va, const vec3& vb, const vec3& vc, const vec3& vna, const vec3& vnb, const vec3& vnc)
	: a(va),
	b(vb),
	c(vc),
	n(),
	na(vna),
	nb(vnb),
	nc(vnc),
	fa(-1),
	fb(-1),
	fc(-1),
	center(),
	perVertexNormal(false),
	selected(0)
{
	center = (a + b + c) / 3.0f;
}

Triangle::Triangle(const vec3& va, const vec3& vb, const vec3& vc, const vec3& vna, const vec3& vnb, const vec3& vnc, const vec2& tan, const vec2& tbn, const vec2& tcn)
	: a(va),
	b(vb),
	c(vc),
	n(),
	na(vna),
	nb(vnb),
	nc(vnc),
	ta(tan),
	tb(tbn),
	tc(tcn),
	fa(-1),
	fb(-1),
	fc(-1),
	center(),
	perVertexNormal(false),
	selected(0)
{
	center = (a + b + c) / 3.0f;
}

Triangle::Triangle(const vec3 &va, const vec3 &vb, const vec3 &vc, const vec3 &vn, int ifa, int ifb, int ifc)
: a(va), 
  b(vb), 
  c(vc), 
  n(vn),
  fa(ifa), 
  fb(ifb),
  fc(ifc), 
  perVertexNormal(false),
  selected(0)
{
    center = (a + b + c) / 3.0f;
}

Triangle::Triangle(const vec3 &va, const vec3 &vb, const vec3 &vc, const vec3 &vn, const vec3 &vna, const vec3 &vnb, const vec3 &vnc, int ifa, int ifb, int ifc)
: a(va), 
  b(vb), 
  c(vc), 
  n(vn),
  na(vna), 
  nb(vnb),
  nc(vnc),
  fa(ifa), 
  fb(ifb),
  fc(ifc), 
  perVertexNormal(false),
  selected(0)
{
    center = (a + b + c) / 3.0f;
}

Triangle::Triangle(const Triangle &t)
: a(t.a), 
  b(t.b),
  c(t.c),
  na(t.na), 
  nb(t.nb),
  nc(t.nc),
  n(t.n),
  neighbors(t.neighbors), 
  fa(t.fa),
  fb(t.fb),
  fc(t.fc), 
  perVertexNormal(t.perVertexNormal),
  selected(t.selected)
{
    center = (a + b + c) / 3.0f;
}

Triangle::~Triangle()
{
}

Triangle &Triangle::operator =(const Triangle &t)
{
    this->a = t.a;
    this->b = t.b;
    this->c = t.c;
    this->n = t.n;

    this->na = t.a;
    this->nb = t.b;
    this->nc = t.c;

    this->perVertexNormal = t.perVertexNormal;

    this->fa = t.fa;
    this->fb = t.fb;
    this->fc = t.fc;

    this->neighbors = t.neighbors;
    this->center = (a + b + c) / 3.0f;

	this->selected = t.selected;

    return *this;
}

bool Triangle::intersect(const vec3 &rayStart, const vec3 &rayDir, float &t) const
{
    return intersect(rayStart, rayDir, a, b, c, t);
}

bool Triangle::intersect(vec3 rayStart, vec3 rayDir, vec3 v0, vec3 v1, vec3 v2, float &t)const
{
    float eps = 0.000000001f;
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;

    vec3 p = cross(rayDir, e2);
    float a = dot(e1, p);

    if (a > -eps && a < eps)
    {
        return false;
    }

    float f = 1.0f / a;
    vec3 s = rayStart - v0;
    float u = f * dot(s, p);

    if (u < 0.0f || u > 1.0f)
        return false;

    vec3 q = cross(s, e1);
    float v = f * dot(rayDir, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * dot(e2, q);

    return true;
}

bool Triangle::intersect(const vec3 &rayStart, const vec3 &rayDir, vec3 &point)const
{
	return intersect(rayStart, rayDir, a, b, c, point);
}

bool Triangle::intersect(vec3 rayStart, vec3 rayDir, vec3 v0, vec3 v1, vec3 v2, vec3 &point)const
{
	float eps = 0.000000001f;
	vec3 u = v1 - v0;
	vec3 v = v2 - v0;
	vec3 n = cross(u, v);
	if (n == vec3(0, 0, 0))
		return false; //triangle is a segment or point
	
	vec3 w0 = rayStart - v0;
	float a = -dot(n, w0);
	float b = dot(n, rayDir);
	if (fabs(b) < eps)
	{
		if (a == 0)
			return false; //same plane
		else
			return false; //disjoint
	}

	float r = a / b;
	if (r < 0.0)
		return false; //no intersect
	
	point = rayStart + r * rayDir;

	float uu, uv, vv, wu, wv, D;

	uu = dot(u, u);
	uv = dot(u, v);
	vv = dot(v, v);
	vec3 w = point - v0;
	wu = dot(w, u);
	wv = dot(w, v);
	D = uv * uv - uu * vv;

	float s, t;
	s = (uv * wv - vv * wu) / D;
	if (s < 0.0 || s > 1.0)         
		return false;
	t = (uv * wu - uu * wv) / D;
	if (t < 0.0 || (s + t) > 1.0) 
		return false;                      

	return true;
}

bool Triangle::inside(const vec3 &p) const
{
    // Prepare our barycentric variables
    vec3 u = b - a;
    vec3 v = c - a;
    vec3 w = p - a;

    vec3 vCrossW = cross(v, w);
    vec3 vCrossU = cross(v, u);

    // Test sign of r
    if (dot(vCrossW, vCrossU) < 0)
        return false;

    vec3 uCrossW = cross(u, w);
    vec3 uCrossV = cross(u, v);

    // Test sign of t
    if (dot(uCrossW, uCrossV) < 0)
        return false;

    // At this point, we know that r and t and both > 0.
    // Therefore, as long as their sum is <= 1, each must be less <= 1
    float denom = length(uCrossV);
    float r = length(vCrossW) / denom;
    float t = length(uCrossW) / denom;

    return (r + t <= 1);
}

vec3 Triangle::getBarycentric(const vec3 &p)
{
    // Christer Ericson's Real-Time Collision Detection 
    //http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates

	vec3 v0 = b - a, v1 = c - a, v2 = p - a;
	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float denom = d00 * d11 - d01 * d01;
	
	vec3 bary = vec3((d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom, 1.0f);
	bary.z -= bary.x;
	bary.z -= bary.y;
	return bary;
}

double Triangle::getArea()
{
	float lengthA = (b - c).length();
	float lengthB = (a - c).length();
	float lengthC = (a - b).length();
	float semiPerimeter = (lengthA + lengthB + lengthC) / 2.0f;
	return sqrt(semiPerimeter * (semiPerimeter - lengthA) * (semiPerimeter - lengthB) * (semiPerimeter - lengthC));
}

int Triangle::getNeighbourCrossing(const vec3 &pbary, vec3 * v0_out, vec3 * v1_out)
{	
	//Select the triangle vertices that belong to the edge we crossed
	vec3 * v0, *v1;		

	if (pbary.x < 0.0f){
		v0 = &a; v1 = &c;
	}
	else if (pbary.y < 0.0f){
		v0 = &a; v1 = &b;
	}
	else{
		v0 = &b; v1 = &c;
	}


	if (v0_out != nullptr) *v0_out = *v0;
	if (v1_out != nullptr) *v1_out = *v1;

	//Determine to which neighbour we crossed
	for (int i = 0; i < neighbors.size(); i++){
		Triangle::Neighbor n = neighbors[i];

		//Check if edge vertices match -> works, but not a good way, todo better, implicitly by index
		if ((n.s == *v0 && n.t == *v1) || (n.t == *v0 && n.s == *v1)){
			return i;
		}
	}

	return -1;
}

int Triangle::getNeighbourCrossingIntersection(const vec3& origin, const vec3& dir, float * tOut, vec3 * edgeNormal, float maxDist)
{
	vec3 v[3]; v[0] = a; v[1] = b; v[2] = c;
	float minT = math_maxfloat;
	int minIndex = -1;		
	float ts[3];	

	vec3 triN = this->n;

	const float eps = 0.000001f;
	for (int i = 0; i < 3; i++){		
		vec3 eVec = v[i] - v[(i+1)%3];		
		vec3 eNormal = normalize(cross(eVec, triN));	

		float t = rayPlaneIntersection(origin, dir, eNormal, v[i]);		

		
		ts[i] = t;
		if (t >= 0.0f && t < minT){
			minT = t;
			minIndex = i;
			*edgeNormal = eNormal;
		}
	}

	
	
	assert(minT <= maxDist);

	assert(minIndex >= 0);
	
	*tOut = minT;

	//Determine to which neighbour we crossed
	for (int i = 0; i < neighbors.size(); i++){
		Triangle::Neighbor n = neighbors[i];

		//Check if edge vertices match -> works, but not a good way, todo better, implicitly by index
		if (
			(n.s == v[minIndex] && n.t == v[(minIndex + 1) % 3]) || 
			(n.t == v[minIndex] && n.s == v[(minIndex + 1) % 3])
		){
			
			return i;
		}
	}
	
	return -1;


}

int Triangle::shareCommonEdge(const Triangle &t, vec3 &s, vec3 &e)
{
    bool asim = false, bsim = false, csim = false;
	const float eps = 0.0001f;

    if (fa == t.fa || fa == t.fb || fa == t.fc ||
		lengthSq(t.a - a) < eps || lengthSq(t.b - a) < eps || lengthSq(t.c - a) < eps
		)
        asim = true;

    if (fb == t.fa || fb == t.fb || fb == t.fc ||
		lengthSq(t.a - b) < eps || lengthSq(t.b - b) < eps || lengthSq(t.c - b) < eps
		)
        bsim = true;

    if (fc == t.fa || fc == t.fb || fc == t.fc ||
		lengthSq(t.a - c) < eps || lengthSq(t.b - c) < eps || lengthSq(t.c - c) < eps
		)
        csim = true;

    int result = -1;

    if (asim && bsim)
    {
        s = a;
        e = b;

        result = 0;
    }        
    
    if (asim && csim)
    {
        s = a;
        e = c;

        result = 1;
    }     

    if (bsim && csim)
    {
        s = b;
        e = c;

        result = 2;
    }

    return result;
}

Plane::Plane( vec3 &v1,  vec3 &v2,  vec3 &v3) 
{
	set3Points(v1,v2,v3);
}

Plane::Plane()
: d(0.0f) 
{
}

Plane::~Plane() {}

void Plane::set3Points( vec3 &v1,  vec3 &v2,  vec3 &v3) 
{
	vec3 aux1, aux2;

	aux1 = v1 - v2;
	aux2 = v3 - v2;

    normal = aux2.cross(aux1);

	normal.normalize();
	point = v2;
	d = -(normal.dot(point));
}

void Plane::setNormalAndPoint(vec3 &normal, vec3 &point) 
{
	this->normal = normal;
	this->normal.normalize();
	d = -(this->normal.dot(point));
}

void Plane::setCoefficients(float a, float b, float c, float d) 
{
	// set the normal vector
	normal.set(a,b,c);
	//compute the lenght of the vector
	float l = normal.length();
	// normalize the vector
	normal.set(a/l,b/l,c/l);
	// and divide d by th length as well
	this->d = d/l;
}

float Plane::distance(vec3 p) 
{
	return (d + normal.dot(p));
}

void Plane::print() 
{
	printf("Plane(");
    normal.print();
    printf("# %f)",d);
}

Frustum::Frustum()
: m_angle (0.0), 
  ntl(), ntr(), nbl(), nbr(), ftl(), ftr(), fbl(), fbr(), 
  nearD(0.0f), farD(0.0f), ratio(0.0f), angle(0.0f), tang(0.0f),
  nw(0.0f), nh(0.0f), fw(0.0f), fh(0.0f)
{
}

Frustum::~Frustum()
{
}

void Frustum::setCamInternals(float angle, float ratio, float nearD, float farD) 
{
	this->ratio = ratio;
	this->angle = angle;
	this->nearD = nearD;
	this->farD = farD;

	tang = (float)tan(angle* math_radians * 0.5) ;
	
    nh = nearD * tang;
	nw = nh * ratio; 

	fh = farD  * tang;
	fw = fh * ratio;
}

void Frustum::setCamDef(vec3 &p, vec3 &l, vec3 &u) 
{
	vec3 dir, nc, fc, X, Y, Z;

	Z = p + (l-p); 
	Z.normalize();

	X = cross(u, Z);
	X.normalize();

	Y = cross(Z, X);

	nc = p - Z * nearD;
	fc = p - Z * farD;

	ntl = nc + Y * nh - X * nw;
	ntr = nc + Y * nh + X * nw;
	nbl = nc - Y * nh - X * nw;
	nbr = nc - Y * nh + X * nw;

	ftl = fc + Y * fh - X * fw;
	ftr = fc + Y * fh + X * fw;
	fbl = fc - Y * fh - X * fw;
	fbr = fc - Y * fh + X * fw;

	pl[TOP].set3Points(ntr, ntl, ftl);
	pl[BOTTOM].set3Points(nbl, nbr, fbr);
	pl[LEFT].set3Points(ntl, nbl, fbl);
	pl[RIGHT].set3Points(nbr, ntr, fbr);
	pl[NEARP].set3Points(ntl, ntr, nbr);
	pl[FARP].set3Points(ftr, ftl, fbl);
}

int Frustum::pointInFrustum(vec3 &p) 
{

	int result = INSIDE;
	for(int i=0; i < 6; i++) {

		if (pl[i].distance(p) < 0)
			return OUTSIDE;
	}
	return(result);

}

int Frustum::sphereInFrustum(vec3 &p, float raio) 
{

	int result = INSIDE;
	float distance;

	for(int i=0; i < 6; i++) {
		distance = pl[i].distance(p);
		if (distance < -raio)
			return OUTSIDE;
		else if (distance < raio)
			result =  INTERSECT;
	}
	return(result);

}

int Frustum::boxInFrustum(vec3 &min, vec3 &max) 
{

	int result = INSIDE;

	for(int i=0; i < 6; i++) 
    {
        vec3 normal = pl[i].normal;

        if(pl[i].distance(getVertexP(normal, min, max)) < 0)
        {
            result = OUTSIDE;            
            return result;
        }
		else if (pl[i].distance(getVertexN(normal, min, max)) < 0)
			result =  INTERSECT;
	}

	return(result);
 }

vec3 Frustum::getVertexP(vec3 &normal,  vec3 &min, vec3 &max) 
{
	vec3 res = min;

	if (normal.x > 0)
		res.x += (max.x - min.x);

	if (normal.y > 0)
		res.y += (max.y - min.y);

	if (normal.z > 0)
		res.z += (max.z - min.z);

	return(res);
}

vec3 Frustum::getVertexN(vec3 &normal,  vec3 &min, vec3 &max) 
{
	vec3 res = min;

	if (normal.x < 0)
		res.x += (max.x - min.x);

	if (normal.y < 0)
		res.y += (max.y - min.y);

	if (normal.z < 0)
		res.z += (max.z - min.z);

	return(res);
}

void Frustum::drawPoints() 
{
	glBegin(GL_POINTS);

		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(nbl.x,nbl.y,nbl.z);
		glVertex3f(nbr.x,nbr.y,nbr.z);

		glVertex3f(ftl.x,ftl.y,ftl.z);
		glVertex3f(ftr.x,ftr.y,ftr.z);
		glVertex3f(fbl.x,fbl.y,fbl.z);
		glVertex3f(fbr.x,fbr.y,fbr.z);

	glEnd();
}

void Frustum::drawLines() 
{
	glBegin(GL_LINE_LOOP);
	//near plane
		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(nbr.x,nbr.y,nbr.z);
		glVertex3f(nbl.x,nbl.y,nbl.z);
	glEnd();

	glBegin(GL_LINE_LOOP);
	//far plane
		glVertex3f(ftr.x,ftr.y,ftr.z);
		glVertex3f(ftl.x,ftl.y,ftl.z);
		glVertex3f(fbl.x,fbl.y,fbl.z);
		glVertex3f(fbr.x,fbr.y,fbr.z);
	glEnd();

	glBegin(GL_LINE_LOOP);
	//bottom plane
		glVertex3f(nbl.x,nbl.y,nbl.z);
		glVertex3f(nbr.x,nbr.y,nbr.z);
		glVertex3f(fbr.x,fbr.y,fbr.z);
		glVertex3f(fbl.x,fbl.y,fbl.z);
	glEnd();

	glBegin(GL_LINE_LOOP);
	//top plane
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(ftl.x,ftl.y,ftl.z);
		glVertex3f(ftr.x,ftr.y,ftr.z);
	glEnd();

	glBegin(GL_LINE_LOOP);
	//left plane
		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(nbl.x,nbl.y,nbl.z);
		glVertex3f(fbl.x,fbl.y,fbl.z);
		glVertex3f(ftl.x,ftl.y,ftl.z);
	glEnd();

	glBegin(GL_LINE_LOOP);
	// right plane
		glVertex3f(nbr.x,nbr.y,nbr.z);
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(ftr.x,ftr.y,ftr.z);
		glVertex3f(fbr.x,fbr.y,fbr.z);

	glEnd();
}

void Frustum::drawPlanes() 
{    
	glBegin(GL_QUADS);	
	glColor4f(0.3f, 0.3f, 0.3f, 0.5f);
    //glColor4f(1.0, 0.0, 0.0, 1.0);
	//near plane
		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(nbr.x,nbr.y,nbr.z);
		glVertex3f(nbl.x,nbl.y,nbl.z);

	//far plane
		//glVertex3f(ftr.x,ftr.y,ftr.z);
		//glVertex3f(ftl.x,ftl.y,ftl.z);
		//glVertex3f(fbl.x,fbl.y,fbl.z);
		//glVertex3f(fbr.x,fbr.y,fbr.z);

    //glColor4f(0.0, 1.0, 0.0, 1.0);

	//bottom plane
		glVertex3f(nbl.x,nbl.y,nbl.z);
		glVertex3f(nbr.x,nbr.y,nbr.z);
		glVertex3f(fbr.x,fbr.y,fbr.z);
		glVertex3f(fbl.x,fbl.y,fbl.z);

	//top plane
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(ftl.x,ftl.y,ftl.z);
		glVertex3f(ftr.x,ftr.y,ftr.z);

        //glColor4f(0.0, 0.0, 1.0, 1.0);

	//left plane

		glVertex3f(ntl.x,ntl.y,ntl.z);
		glVertex3f(nbl.x,nbl.y,nbl.z);
		glVertex3f(fbl.x,fbl.y,fbl.z);
		glVertex3f(ftl.x,ftl.y,ftl.z);


	// right plane
		glVertex3f(nbr.x,nbr.y,nbr.z);
		glVertex3f(ntr.x,ntr.y,ntr.z);
		glVertex3f(ftr.x,ftr.y,ftr.z);
		glVertex3f(fbr.x,fbr.y,fbr.z);

	glEnd();
}

void Frustum::drawNormals() 
{
	vec3 a,b;

	glBegin(GL_LINES);

		// near
		a = (ntr + ntl + nbr + nbl) * 0.25;
		b = a + pl[NEARP].normal;
		glVertex3f(a.x,a.y,a.z);
		glVertex3f(b.x,b.y,b.z);

		// far
		a = (ftr + ftl + fbr + fbl) * 0.25;
		b = a + pl[FARP].normal;
		glVertex3f(a.x,a.y,a.z);
		glVertex3f(b.x,b.y,b.z);

		// left
		a = (ftl + fbl + nbl + ntl) * 0.25;
		b = a + pl[LEFT].normal;
		glVertex3f(a.x,a.y,a.z);
		glVertex3f(b.x,b.y,b.z);
		
		// right
		a = (ftr + nbr + fbr + ntr) * 0.25;
		b = a + pl[RIGHT].normal;
		glVertex3f(a.x,a.y,a.z);
		glVertex3f(b.x,b.y,b.z);
		
		// top
		a = (ftr + ftl + ntr + ntl) * 0.25;
		b = a + pl[TOP].normal;
		glVertex3f(a.x,a.y,a.z);
		glVertex3f(b.x,b.y,b.z);
		
		// bottom
		a = (fbr + fbl + nbr + nbl) * 0.25;
		b = a + pl[BOTTOM].normal;
		glVertex3f(a.x,a.y,a.z);
		glVertex3f(b.x,b.y,b.z);

	glEnd();
}

Picking::Picking()
{
}

Picking::~Picking()
{
}

vec3 Picking::pickingRay(Camera* camera, int mouse_x, int mouse_y, float window_width, float window_height)
{
	double aspect = double(window_width)/double(window_height);
	Frustum* frustum = camera->frustum();
	
	float near_height = frustum->nh;

	int window_y = int(window_height - mouse_y) - int(window_height/2.0f); 
	float norm_y = float(window_y) / float(window_height / 2.0f);
	int window_x = int((mouse_x - window_width) / 2.0f); 
	float norm_x = float(window_x)/ float(window_width/2.0f);

	float y = near_height * norm_y; 
	float x = float(near_height * aspect * norm_x);

	float zNear = frustum->nearD;

	return vec3(x, y, -zNear);
}

void Picking::getPickingRay(const Transform &trans, const float fov, const float ncp, const float window_width, const float window_height, float mouse_x, float mouse_y, vec3 &rayPos, vec3 &rayDir)
{
    //http://schabby.de/picking-opengl-ray-tracing/

	vec3 camPos  = getCamPosFromModelView(trans);
	vec3 viewDir = getViewDirFromModelView(trans);
	vec3 upDir   = getUpDirFromModelView(trans);

	double aspect = double(window_width)/double(window_height);

	vec3 view = normalize(viewDir);
    
    vec3 tmpa = cross(view, upDir);
	vec3 h = normalize(tmpa);
    
    vec3 tmpb = cross(h, view);
	vec3 v = normalize(tmpb);

	float rad = fov * math_radians;
	float vLength = float(tan( rad / 2.0 ) * ncp);
	float hLength = vLength * (window_width / window_height);

	v *= vLength;
	h *= hLength;

	// translate mouse coordinates so that the origin lies in the center of the view port
	mouse_y = window_height - mouse_y;

	mouse_x -= window_width / 2.0f;
	mouse_y -= window_height / 2.0f;

	// scale mouse coordinates so that half the view port width and heigh becomes 1
	mouse_y /= (window_height / 2.0f);
	mouse_x /= (window_width / 2.0f);

	// linear combination to compute intersection of picking ray with view port plane
	vec3 pos = camPos + view*ncp + h*mouse_x + v*mouse_y;
    vec3 tmp = pos - camPos;
	vec3 dir = normalize(tmp);

	rayPos = camPos;
	rayDir = dir;
}

vec3 Picking::intersectionPlane(vec3 v1, vec3 v2, vec3 v3, vec3 startPos, vec3 endPos)
{
	//vec3 n = (v3-v2).cross(v1-v2);
	//float b  =  v2.dot(n);
	//float length = n.length();


	//Vector3 s;
	//Vector3 dir = endPos - startPos;
	//float angle = n.angle(dir);

	//float dist_start = abs((n.dot(startPos) - b) / length);
	//float dist_end   = abs((n.dot(endPos) - b) / length);

	//if(angle != math_pi/2.0f)
	//{   // not parallel
	//		float b = v2.x*n.x + v2.y*n.y + v2.z*n.z;
	//		float tmp = n.dot(dir);

	//		if(tmp != 0.0f)
	//		{
	//			float t = (b - n.dot(startPos)) / tmp ;
	//			if(0.0f<= t && t <=1.0f)
	//			{

	//				if(dist_end < dist_start)
	//				{
	//					s = startPos + t*dir;	
	//				}
	//			}

	//		}
	//}

	//return s;

	vec3 n = (v3-v2).cross(v1-v2);
	float b  =  v2.dot(n);
	float length = n.length();

	Vector3 s;
	Vector3 dir = endPos - startPos;
	float angle = n.angle(dir);

	float dist_start = std::abs((n.dot(startPos) - b) / length);
	float dist_end   = std::abs((n.dot(endPos) - b) / length);

	if(angle != math_pi/2.0f)
	{   // not parallel
		float b = v2.x*n.x + v2.y*n.y + v2.z*n.z;
		float tmp = n.dot(dir);

		if(tmp != 0.0f)
		{
			float t = (b - n.dot(startPos)) / tmp ;
			//if(0.0f<= t && t <=1.0f)
			{

				//if(dist_end < dist_start)
				{
					s = startPos + t*dir;	

				}
			}
		}
	}

	return s;
}

bool Picking::intersectTriangle(vec3 rayStart, vec3 rayDir, vec3 v0, vec3 v1, vec3 v2, float &t)
{
	float eps = 0.000000001f;

	vec3 e1 = v1 - v0;
	vec3 e2 = v2 - v0;

	vec3 p = cross(rayDir, e2);
	float a = dot(e1, p);

	if(a > -eps && a < eps)
	{
		return false;
	}

	float f = 1.0f / a;
	vec3 s = rayStart - v0;
	float u = f * dot(s,p);

	if(u < 0.0f || u > 1.0f)
		return false;

	vec3 q = cross(s, e1);
	float v = f * dot(rayDir, q);

	if(v < 0.0f || u+v > 1.0f)
		return false;

	t = f * dot(e2, q);

	return true;
}

bool Picking::intersectQuad(vec3 rayStart, vec3 rayDir, vec3 v0, vec3 v1, vec3 v2, vec3 v3, float &t)
{
	float t1 = math_maxfloat;
	float t2 = math_maxfloat;

	bool b1 = intersectTriangle(rayStart, rayDir, v3, v1, v2, t1);
	bool b2 = intersectTriangle(rayStart, rayDir, v3, v0, v1, t2);

	t = g_min<float>(t1,t2);
	return b1 || b2;	
}

bool Picking::intersectCube(vec3 rayStart, vec3 rayDir, vec3 v0, vec3 v1, vec3 v2, vec3 v3, vec3 v4, vec3 v5, vec3 v6, vec3 v7, float &t)
{
	t = math_maxfloat;
	float s = math_maxfloat;
	bool b0, b1, b2, b3, b4, b5;

	// back
	if(b0 = intersectQuad(rayStart, rayDir, v0, v1, v2, v3, s)) 
		t =  g_min<float>(t, s);

	//front
	if(b1 = intersectQuad(rayStart, rayDir, v4, v5, v6, v7, s))
		t = g_min<float>(t, s);
	

	// left
	if(b2 = intersectQuad(rayStart, rayDir, v0, v4, v7, v3, s))
		t = g_min<float>(t, s);
	
	// right
	if(b3 = intersectQuad(rayStart, rayDir, v5, v1, v2, v6, s))
		t = g_min<float>(t, s);
	
	// top
	if(b4 = intersectQuad(rayStart, rayDir, v7, v6, v2, v3, s))
		t = g_min<float>(t, s);


	// bottom
	if(b5 = intersectQuad(rayStart, rayDir, v4, v5, v1, v0, s))
		t = g_min<float>(t, s);

	return b0 || b1 || b2 || b3 || b4 || b5;
}

float Picking::select(const Transform &trans, const mat4 &matModel, const vec3 &mi, const vec3 &ma, int width, int height, int mx, int my)
{
    vec3 rayPnt;
	vec3 rayDir;

	getPickingRay(trans, params::inst()->fov, params::inst()->ncp, (float)width, (float)height, (float)mx, (float)my, rayPnt, rayDir);

    vec4 tmpMa = matModel * vec4(ma.x, ma.y, ma.z, 1);
    vec4 tmpMi = matModel * vec4(mi.x, mi.y, mi.z, 1);

    //tmpMa = mat4::translate(o->m_position) * mat4::rotateY(o->m_rotation.y) * mat4::scale(o->m_scale) * tmpMa;
    //tmpMi = mat4::translate(o->m_position) * mat4::rotateY(o->m_rotation.y) * mat4::scale(o->m_scale) * tmpMi;    

    vec3 mma = vec3(tmpMa.x, tmpMa.y, tmpMa.z);
    vec3 mmi = vec3(tmpMi.x, tmpMi.y, tmpMi.z);

    vec3 v0 = vec3(mmi.x, mmi.y, mmi.z);
    vec3 v1 = vec3(mma.x, mmi.y, mmi.z);
    vec3 v2 = vec3(mma.x, mma.y, mmi.z);
    vec3 v3 = vec3(mmi.x, mma.y, mmi.z);

    vec3 v4 = vec3(mmi.x, mmi.y, mma.z);
    vec3 v5 = vec3(mma.x, mmi.y, mma.z);
    vec3 v6 = vec3(mma.x, mma.y, mma.z);
    vec3 v7 = vec3(mmi.x, mma.y, mma.z);

    float t = 0;

    if(intersectCube(rayPnt, rayDir, v0, v1, v2, v3, v4, v5, v6, v7, t))
    {
        return t;
    }

    return -1.0f;
}

Spline::Spline(Config conf) 
: m_config(conf),
  m_deltaT(0.0f),
  m_phantomStart(),
  m_phantomEnd()
{
}

Spline::~Spline()
{
}

void Spline::addPoint(const vec3 &v)
{
    m_points.push_back(v);
    m_deltaT = (float)1 / ((float)m_points.size()-1);

    if(m_points.size() >= 3)
    {
        vec3 sStart = m_points[0];
        vec3 tStart = m_points[1];

        vec3 nStart = vec3(sStart - tStart);
        float lenStart = nStart.length();
        m_phantomStart = sStart +  nStart.normalized() * lenStart * 0.5;

        uint size = (uint)m_points.size() - 1;
        vec3 sEnd = m_points[size];
        vec3 tEnd = m_points[size - 1];

        vec3 nEnd = vec3(sEnd - tEnd);
        float lenEnd = nEnd.length();
        m_phantomEnd = sEnd +  nEnd.normalized() * lenEnd * 0.5;
    }
}

void Spline::bounds(int &p)
{
    if (p < 0) 
        p = 0; 

    else if (p >= (int)m_points.size()-1) 
        p = int(m_points.size() - 1);
}

vec3 Spline::interpolatedPoint(float t, Config conf)
{    
    // Find out in which interval we are on the spline
    int p = (int)(t / m_deltaT);

    int p0 = p - 1; 
    int p1 = p;  
    int p2 = p + 1;
    int p3 = p + 2; 

    vec3 c1;
    if(p0 < 0)
        c1 = m_phantomStart;
    else
        c1 = m_points[p0];

    vec3 v1 = m_points[p1];
    vec3 v2;
    if((uint)p2 >= m_points.size())
        v2 = m_points[m_points.size()-1];    
    else
        v2 = m_points[p2];

    vec3 c2;
    if((uint)p3 >= m_points.size()-1)
        c2 = m_phantomEnd;
    else
        c2 = m_points[p3];

    // Relative (local) time 
	float lt = (t - m_deltaT * (float)p) / m_deltaT;

    if(conf == CATMULL_ROM)
        return Spline::catmullRomInterpolation(c1, v1, v2, c2, lt);
    if(conf == CUBIC)
        return Spline::bSplineInterpolation(c1, v1, v2, c2, lt);
    if(conf == BSPLINE)
        return Spline::cubicInterpolation(c1, v1, v2, c2, lt);
    if(conf == HERMITE)
        return Spline::hermiteInterpolation(c1, v1, v2, c2, lt);
    if(conf == KOCHANEK_BARTEL)
        return Spline::kochanekBartelInterpolation(c1, v1, v2, c2, lt);
    if(conf == ROUNDED_CATMULL_ROM)
        return Spline::roundedCatmullRomInterpolation(c1, v1, v2, c2, lt);

	//return LINEAR in default case
	return Spline::linearInterpolation(v1, v2, lt);
}

vec3 Spline::point(int n) const
{
    return m_points[n];
}

int Spline::numPoints() const
{
    return (int)m_points.size();
}

void Spline::render(Config conf)
{
    glPushMatrix();

        //Interpolated Points
        glColor3f(0.7f, 0.7f, 0.7f);
        glPointSize(1.0f);
        glBegin(GL_POINTS);

        for(float f=0; f<1.0f; f+= 0.005f/m_points.size())
        {
            vec3 v = interpolatedPoint(f, conf);
            glVertex3f(v.x, v.y, v.z);
        }        
        glEnd();

        //Base Points
        glColor3f(1.0f, 1.0f, 1.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);

        for(uint i=0; i<m_points.size(); ++i)
        {
            vec3 v = m_points[i];
            glVertex3f(v.x, v.y, v.z);
        }        
        glEnd();


        //Phantom Points
        vec3 p1 = m_phantomStart;
        vec3 p2 = m_phantomEnd;

        glPointSize(3.0f);
        glBegin(GL_POINTS);        
            glColor3f(1.0f, 1.0f, 0.0f);    
            glVertex3f(p1.x, p1.y, p1.z);
            glColor3f(0.0f, 1.0f, 1.0f);                    
            glVertex3f(p2.x, p2.y, p2.z);
        glEnd();

        //Start/End for Phoantom Points
        uint size = (uint)m_points.size() - 1;
        vec3 sStart = m_points[0];
        vec3 tStart = m_points[1];
        vec3 sEnd   = m_points[size];
        vec3 tEnd   = m_points[size-1];   

        glPointSize(8.0f);
        glBegin(GL_POINTS);
            glColor3f(1.0f, 0.0f, 0.0f);    
            glVertex3f(sStart.x, sStart.y, sStart.z);
            glColor3f(0.5f, 0.0f, 0.0f);    
            glVertex3f(tStart.x, tStart.y, tStart.z);

            glColor3f(0.0f, 0.0f, 1.0f);    
            glVertex3f(sEnd.x, sEnd.y, sEnd.z);
            glColor3f(0.0f, 0.0f, 0.5f);    
            glVertex3f(tEnd.x, tEnd.y, tEnd.z);
        glEnd();
    glPopMatrix();
}

void Spline::clear()
{
    m_points.clear();
}

vec3 Spline::linearInterpolation(const vec3 &p0, const vec3 &p1, float t)
{
    return p0 * t + (p1 * (1-t));
}

vec3 Spline::catmullRomInterpolation(const vec3 &p0, const vec3 &p1, const Vector3 &p2, const vec3 &p3, float t)
{
    float t2 = t * t;
    float t3 = t2 * t;

    float b1 = 0.5f * (  -t3 + 2*t2 - t);
    float b2 = 0.5f * ( 3*t3 - 5*t2 + 2);
    float b3 = 0.5f * (-3*t3 + 4*t2 + t);
    float b4 = 0.5f * (   t3 -   t2    );

    return (p0*b1 + p1*b2 + p2*b3 + p3*b4); 
}

vec3 Spline::roundedCatmullRomInterpolation(const vec3 &p0, const vec3 &p1, const vec3 &p2, const vec3 &p3, float t)
{
    //not working right

    vec3 ab = p0 - p1;
    vec3 cb = p2 - p3;

    vec3 bVelo = cb.normalized() - ab.normalized();
    bVelo.normalize();

    vec3 dc = p3 - p2;
    vec3 bc = -cb;   

    vec3 cVelo = dc.normalized() - bc.normalized();
    cVelo.normalize();

    float cbDist = cb.length();

    return Spline::catmullRomInterpolation(bVelo * cbDist, p1, p2, cVelo * cbDist, t);
}

vec3 Spline::cubicInterpolation(const vec3 &p0, const vec3 &p1, const vec3 &p2, const vec3 &p3, float t)
{
   float t2;
   vec3 a0, a1, a2, a3;

   t2 = t*t;
   a0 = p3 - p2 - p0 + p1;
   a1 = p0 - p1 - a0;
   a2 = p2 - p0;
   a3 = p1;

   return(a0*t*t2 + a1*t2+a2*t + a3);
}

vec3 Spline::bSplineInterpolation(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &p4, float t)
{
    //not working right

    Vector4 a, b, c;
    
    a.x = (-p1.x + 3.0f * p2.x - 3.0f * p3.x + p4.x) / 6.0f;
    a.y = (3.0f * p1.x - 6.0f * p2.x + 3.0f * p3.x) / 6.0f;
    a.z = (-3.0f * p1.x + 3.0f * p3.x) / 6.0f;
    a.w = (p1.x + 4.0f * p2.x + p3.x) / 6.0f;
    
    b.x = (-p1.y + 3 * p2.y - 3.0f * p3.y + p4.y) / 6.0f;
    b.y = (3.0f * p1.y - 6.0f * p2.y + 3.0f * p3.y) / 6.0f;
    b.z = (-3.0f * p1.y + 3.0f * p3.y) / 6.0f;
    b.w = (p1.y + 4 * p2.y + p3.y) / 6.0f;

    c.x = (-p1.y + 3.0f * p2.y - 3.0f * p3.y + p4.y) / 6.0f;
    c.y = (3.0f * p1.y - 6.0f * p2.y + 3.0f * p3.y) / 6.0f;
    c.z = (-3.0f * p1.y + 3.0f * p3.y) / 6.0f;
    c.w = (p1.y + 4.0f * p2.y + p3.y) / 6.0f;

    vec3 p = vec3( ((a.z + t * (a.y + t * a.x))*t+a.w), ((b.z + t * (b.y + t * b.x))*t+b.w), ((c.z + t * (c.y + t * c.x))*t+c.w) ) ;
    
    return p;

}

vec3 Spline::hermiteInterpolation(const vec3 &p0, const vec3 &p1, const vec3 &p2, const vec3 &p3, float t, float tension, float bias)
{
    vec3 m0, m1;
    float t2,t3;
    float a0, a1, a2, a3;

    t2 = t * t;
    t3 = t2 * t;

    m0  = (p1-p0)*(1+bias)*(1-tension)/2;
    m0 += (p2-p1)*(1-bias)*(1-tension)/2;
    m1  = (p2-p1)*(1+bias)*(1-tension)/2;
    m1 += (p3-p2)*(1-bias)*(1-tension)/2;
    
    a0 =  2*t3 - 3*t2 + 1;
    a1 =    t3 - 2*t2 + t;
    a2 =    t3 -   t2;
    a3 = -2*t3 + 3*t2;

    return(a0*p1 + a1*m0 + a2*m1 + a3*p2);
}

vec3 Spline::picewiseHermiteInterpolation(const vec3 &a, const vec3 &b, const vec3 &startTangent, const vec3 &endTangent, float t)
{
    float t2, t3;
    float a0, a1, b0, b1;

    t2 = t*t;
    t3 = t2*t;

    a0 = (t3 * 2.0f) - (3.0f * t2) + 1.0f;
    a1 = (-2.0f * t3) + (3.0f * t2);    
    b0 = t3 - (2.0f * t2) + t;
    b1 = t3 -   t2;

    return (a0*a + a1*b + b0*startTangent + b1*endTangent);
}

vec3 Spline::kochanekBartelInterpolation(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &d, float t, float tension, float bias, float continuity)
{
    //not working right

    vec3 ab = vec3(b-a).normalized();
    vec3 cd = vec3(d-c).normalized();

    vec3 inTangent = ((1.0f - tension) * (1.0f - continuity) * (1.0f + bias)) * 0.5f * ab + ((1.0f - tension) * (1.0f + continuity) * (1.0f - bias)) * 0.5f * cd;

    vec3 outTangent = ((1.0f - tension) * (1.0f + continuity) * (1.0f + bias)) * 0.5f * ab + ((1.0f - tension) * (1.0f - continuity) * (1.0f - bias)) * 0.5f * cd;

    //return picewiseHermiteInterpolation(b, c, inTangent, outTangent, t);
    return hermiteInterpolation(inTangent, b, c, outTangent, t);
}

int shareCommonEdge(Face &f1, Face &f2)
{
    bool asim = false, bsim = false, csim = false;

    if (f1.a == f2.a || f1.a == f1.b || f1.a == f1.c)
        asim = true;

    if (f1.b == f2.a || f1.b == f1.b || f1.b == f1.c)
        bsim = true;

    if (f1.c == f2.a || f1.c == f1.b || f1.c == f1.c)
        csim = true;

    int result = -1;

    if (asim && bsim)
        result = 0;
    if (asim && csim)
        result = 1;
    if (bsim && csim)
        result = 2;

    return result;
}

float rayPlaneIntersection(const vec3 &origin, const vec3 &dir, const vec3& planeN, const vec3& planePt)
{
	float denom = dot(dir, planeN);

	const float eps = math_epsilon;
	if (abs(denom) < eps){
		return -1.0f;
	}
	
	return dot((planePt - origin), planeN) / denom;
}

vec3 projectVectorOnPlane(const vec3 &u, const vec3 &n)
{
	float duv = dot(u, n);
	float nl = length(n) * length(n);

	vec3 p = u - duv / nl * n;
	return p;
}

float pointLineDistance(const vec3 &a, const vec3 &b, const vec3 &p)
{
	vec3 pb = p - b;
	vec3 ba = b - a;

	float nom = length(cross(pb, ba));
	float den = length(p - b);

	float dist = nom / den;

	return dist;
}

bool isInsideMesh(const std::vector<Triangle> &triangles, const vec3 &point)
{	
	vec3 dir = vec3(0,1,0);
	int intersections = 0;

	for (int i = 0; i < triangles.size(); ++i)
	{
		const Triangle &t = triangles[i];

		vec3 tmp = vec3(0, 0, 0);
		if (t.intersect(point, dir, tmp)) 
        {
			intersections++;
		}
	}

	if (intersections % 2 == 1) 
    {
		// inside
		return true;
	}
	else {
		// outside
		return false;
	}
}

void normalizeGeometry(std::vector<Vertex> &vertices, const vec3 &translate, const vec3 &scale, const vec4 &rotate)
{
	vec3 mi = vec3(math_maxfloat, math_maxfloat, math_maxfloat);
	vec3 ma = vec3(math_minfloat, math_minfloat, math_minfloat);

	for (int i = 0; i<vertices.size(); ++i)
	{
		vec3 &a = vertices[i].position;

		if (a.x > ma.x) ma.x = a.x;
		if (a.y > ma.y) ma.y = a.y;
		if (a.z > ma.z) ma.z = a.z;

		if (a.x < mi.x) mi.x = a.x;
		if (a.y < mi.y) mi.y = a.y;
		if (a.z < mi.z) mi.z = a.z;
	}
	vec3 d = ma - mi;
	float s = max(d.x, max(d.y, d.z));

	vec3 shift = d / s /2;
	for (int i = 0; i<vertices.size(); ++i)
	{
		vec3 &a = vertices[i].position;

		a -= mi;
		a /= s;
		a -= vec3(shift.x, 0.0f, shift.z);

		mat4 m = mat4::identitiy();
		m *= mat4::translate(translate);
		m *= mat4::rotate(rotate.x, vec3(rotate.y, rotate.z, rotate.w));
		m *= mat4::scale(scale);

		vec4 ta = m * vec4(a);
		vertices[i].position = vec3(ta.x, ta.y, ta.z);
	}

	//mi = vec3(math_maxfloat, math_maxfloat, math_maxfloat);
	//ma = vec3(math_minfloat, math_minfloat, math_minfloat);

	//for(int i=0; i<vertices.size(); ++i)
	//{
	//    vec3 &a = vertices[i].pos;

	//	if (a.x > ma.x) ma.x = a.x;
	//	if (a.y > ma.y) ma.y = a.y;
	//	if (a.z > ma.z) ma.z = a.z;

	//	if (a.x < mi.x) mi.x = a.x;
	//	if (a.y < mi.y) mi.y = a.y;
	//	if (a.z < mi.z) mi.z = a.z;
	//}

	//mi.print();
	//ma.print();
}