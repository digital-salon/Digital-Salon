#include "Mesh.h"
#include "VertexBufferObject.h"



Mesh::Mesh()
{
}

Mesh::~Mesh()
{
}

//vector<VertexBufferObject *> Mesh::obj(const QString &fileName, const vec3 &rot, const vec3 &scale, GLenum primitive)
//{
//    vector<VertexBufferObject *> vbos;

 //   ModelOBJ *model = new ModelOBJ();
	//std::vector<vec3> tempVertices;	

 //   std::cout << "OBSTACLEOBJ::load():" << fileName.toStdString();
 //   model->import(fileName.toStdString().c_str());
 //   
	//uint nrMeshes = model->getNumberOfMeshes();

 //   const ModelOBJ::Vertex *vb = model->getVertexBuffer();
 //   const int *tempIdx = model->getIndexBuffer();

 //   mat4 rotMatX = mat4::rotate(rot.x, 1.0f, 0.0f, 0.0f);
	//mat4 rotMatY = mat4::rotate(rot.y, 0.0f, 1.0f, 0.0f);
 //   mat4 rotMatZ = mat4::rotate(rot.z, 0.0f, 0.0f, 1.0f);
 //   mat4 scaleMat = mat4::scale(scale.x, scale.y, scale.z);

	//for(uint i = 0; i < nrMeshes; ++i)	
	//{
	//	std::vector<vec3> tempVertices;	
	//	std::vector<vec3> tmpNormals;
	//	std::vector<vec3> tmpTexCoords;

	//	const ModelOBJ::Mesh &objMesh = model->getMesh(i);
	//	int startIndex = objMesh.startIndex;
	//	int vertexCount = objMesh.triangleCount * 3;

	//	for(int j=startIndex; j<startIndex + vertexCount; ++j)
	//	{
	//		uint idx = tempIdx[j];
	//		Vector3 v(vb[idx].position[0], vb[idx].position[1], vb[idx].position[2]);
	//		Vector3 n(vb[idx].normal[0], vb[idx].normal[1], vb[idx].normal[2]);
	//		Vector3 t(vb[idx].texCoord[0], vb[idx].texCoord[1], 0.0f);            

	//		tempVertices.push_back(v);
	//		tmpNormals.push_back(n);
	//		tmpTexCoords.push_back(t);
	//	} 

 //       vec3 mi(math_maxfloat, math_maxfloat, math_maxfloat);
 //       vec3 ma(math_minfloat, math_minfloat, math_minfloat);
 //     

 //       VertexBufferObject::DATA *data = new VertexBufferObject::DATA[tempVertices.size()];

	//	for(uint i=0; i<tempVertices.size(); ++i)
	//	{
	//		VertexBufferObject::DATA d;

 //           vec3 v = tempVertices[i];
	//		vec3 n = tmpNormals[i];
	//		vec3 t = tmpTexCoords[i];

	//		d.vx = v.x;
	//		d.vy = v.y;
	//		d.vz = v.z;
	//		d.vw = 1.0f;

	//		d.cx = 1.0f;
	//		d.cy = 0.0f;
	//		d.cz = 0.0f;
	//		d.cw = 1.0;

	//		d.nx = n.x;
	//		d.ny = n.y;
	//		d.nz = n.z;
	//		d.nw = 1.0f;

	//		d.tx = t.x;
	//		d.ty = t.y;
 //           d.tz = 0.0f;
 //           d.tw = 1.0f;

	//		data[i] = d;
	//	}

	//	VertexBufferObject* vboMesh = new VertexBufferObject();
	//	vboMesh->setData(data, GL_STATIC_DRAW, tempVertices.size(), primitive); 

	//	delete[] data;

 //       vbos.push_back(vboMesh);
 //   }

    //return vbos;
//}

//vector<VertexBufferObject*> loadObj(QString fileName)
//{
  //  vector <VertexBufferObject*> vbos;

    //ModelOBJ *model = new ModelOBJ();
    //std::vector<vec3> tempVertices;

    //std::cout << "OBSTACLEOBJ::load():" << fileName.toStdString() << std::endl;
    //model->import(fileName.toStdString().c_str());

    //uint nrMeshes = model->getNumberOfMeshes();

    //const ModelOBJ::Vertex *vb = model->getVertexBuffer();
    //const int *tempIdx = model->getIndexBuffer();


    //vec3 scale(10, 10, 10);
    //vec3 rot(0, 0, 0);

    //mat4 rotMatX = mat4::rotate(rot.x, 1.0f, 0.0f, 0.0f);
    //mat4 rotMatY = mat4::rotate(rot.y, 0.0f, 1.0f, 0.0f);
    //mat4 rotMatZ = mat4::rotate(rot.z, 0.0f, 0.0f, 1.0f);
    //mat4 scaleMat = mat4::scale(scale.x, scale.y, scale.z);

    //for (uint i = 0; i < nrMeshes; ++i)
    //{
    //    std::vector<vec3> tempVertices;
    //    std::vector<vec3> tmpNormals;
    //    std::vector<vec3> tmpTexCoords;

    //    const ModelOBJ::Mesh &objMesh = model->getMesh(i);
    //    int startIndex = objMesh.startIndex;
    //    int vertexCount = objMesh.triangleCount * 3;

    //    for (int j = startIndex; j<startIndex + vertexCount; ++j)
    //    {
    //        uint idx = tempIdx[j];
    //        Vector3 v(vb[idx].position[0], vb[idx].position[1], vb[idx].position[2]);
    //        Vector3 n(vb[idx].normal[0], vb[idx].normal[1], vb[idx].normal[2]);
    //        Vector3 t(vb[idx].texCoord[0], vb[idx].texCoord[1], 0.0f);

    //        tempVertices.push_back(v);
    //        tmpNormals.push_back(n);
    //        tmpTexCoords.push_back(t);
    //    }


    //    VertexBufferObject::DATA *data = new VertexBufferObject::DATA[tempVertices.size()];

    //    for (uint i = 0; i<tempVertices.size(); ++i)
    //    {
    //        VertexBufferObject::DATA d;
    //        vec3 pos = tempVertices[i];
    //        vec3 n = tmpNormals[i];
    //        vec3 t = tmpTexCoords[i];

    //        d.vx = pos.x;
    //        d.vy = pos.y;
    //        d.vz = pos.z;
    //        d.vw = 1.0f;

    //        d.cx = 1.0f;
    //        d.cy = 0.0f;
    //        d.cz = 0.0f;
    //        d.cw = 1.0;

    //        d.nx = n.x;
    //        d.ny = n.y;
    //        d.nz = n.z;
    //        d.nw = 1.0f;

    //        d.tx = t.x;
    //        d.ty = t.y;
    //        d.tz = 0.0f;
    //        d.tw = 1.0f;

    //        data[i] = d;
    //    }

    //    VertexBufferObject* vboMesh = new VertexBufferObject();
    //    vboMesh->setData(data, GL_STATIC_DRAW, tempVertices.size(), GL_TRIANGLES);

    //    delete[] data;

    //    vbos.push_back(vboMesh);
    //}

 //   return vbos;
//}

VertexBufferObject *Mesh::quadLines(int startX, int startY, int width, int height, const vec4 &color)
{
	vec3 mi(float(startX), float(startY), 0.0f);
	vec3 ma(float(startX + width), float(startY + height), 0.0f);

    vector<vec3> vertices;
    vector<vec3> normals;
	vector<vec3> texCoords;

	float d = 0.1f;

    vertices.push_back(vec3(mi.x, mi.y, d));
    vertices.push_back(vec3(mi.x, ma.y, d));

	vertices.push_back(vec3(mi.x, ma.y, d));
    vertices.push_back(vec3(ma.x, ma.y, d));

	vertices.push_back(vec3(ma.x, ma.y, d));
    vertices.push_back(vec3(ma.x, mi.y, d));

	vertices.push_back(vec3(ma.x, mi.y, d));
	vertices.push_back(vec3(mi.x, mi.y, d));
	

   uint nrVertices = (uint)vertices.size();
   VertexBufferObject::DATA *attrData = new VertexBufferObject::DATA[nrVertices];

    for(uint i=0; i<nrVertices; ++i)
    {    
        vec3 v = vertices[i];

        attrData[i].vx = v.x;
        attrData[i].vy = v.y;
        attrData[i].vz = v.z;
        attrData[i].vw = 1.0f;

        attrData[i].nx = 0.0f;
        attrData[i].ny = 0.0f;
        attrData[i].nz = 0.0f;
        attrData[i].nw = 0.0f;

        attrData[i].cx = color.x;
        attrData[i].cy = color.y;
        attrData[i].cz = color.z;
        attrData[i].cw = color.w;

        attrData[i].tx = 0.0f;
        attrData[i].ty = 0.0f;
        attrData[i].tz = 0.0f;
		attrData[i].tw = 0.0f;
    }

    VertexBufferObject *vbo = new VertexBufferObject();
    vbo->setData(attrData, GL_STATIC_DRAW, nrVertices, GL_LINES);

    delete[] attrData;    

    return vbo;
}

VertexBufferObject *Mesh::quad(int startX, int startY, int width, int height, const vec4 &color, GLenum primitive)
{
	vec3 mi(float(startX), float(startY), 0.0f);
	vec3 ma(float(startX + width), float(startY + height), 0.0f);

    vector<vec3> vertices;
    vector<vec3> normals;
	vector<vec3> texCoords;

	float d = 0.1f;

    vertices.push_back(vec3(mi.x, mi.y, d));
    vertices.push_back(vec3(mi.x, ma.y, d));
    vertices.push_back(vec3(ma.x, ma.y, d));
    vertices.push_back(vec3(ma.x, mi.y, d));

    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));

	texCoords.push_back(vec3(0.0f, 0.0f, 0.0f));
	texCoords.push_back(vec3(0.0f, 1.0f, 0.0f));
	texCoords.push_back(vec3(1.0f, 1.0f, 0.0f));
	texCoords.push_back(vec3(1.0f, 0.0f, 0.0f));


   uint nrVertices = (uint)vertices.size();
   VertexBufferObject::DATA *attrData = new VertexBufferObject::DATA[nrVertices];

    for(uint i=0; i<nrVertices; ++i)
    {    
        vec3 v = vertices[i];
        vec3 n = normals[i];
		vec3 t = texCoords[i];

        attrData[i].vx = v.x;
        attrData[i].vy = v.y;
        attrData[i].vz = v.z;
        attrData[i].vw = 1.0f;

        attrData[i].nx = n.x;
        attrData[i].ny = n.y;
        attrData[i].nz = n.z;
        attrData[i].nw = 0.0f;

        attrData[i].cx = color.x;
        attrData[i].cy = color.y;
        attrData[i].cz = color.z;
        attrData[i].cw = color.w;

        attrData[i].tx = t.x;
        attrData[i].ty = t.y;
        attrData[i].tz = 0.0f;
		attrData[i].tw = 0.0f;
    }

    VertexBufferObject *vbo = new VertexBufferObject();
    vbo->setData(attrData, GL_STATIC_DRAW, nrVertices, primitive);
    vbo->bindDefaultAttribs();

    delete[] attrData;    

    return vbo;
}

VertexBufferObject *Mesh::quad(int width, int height, const vec4 &color, GLenum primitive)
{
	vec3 mi(-width/2.0f, -height/2.0f, 0.0f);
	vec3 ma(width/2.0f, height/2.0f, 0.0f);

    vector<vec3> vertices;
    vector<vec3> normals;
	vector<vec3> texCoords;

    vertices.push_back(vec3(mi.x, mi.y, 0.0f));
    vertices.push_back(vec3(mi.x, ma.y, 0.0f));
    vertices.push_back(vec3(ma.x, ma.y, 0.0f));
    vertices.push_back(vec3(ma.x, mi.y, 0.0f));

    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));

	texCoords.push_back(vec3(0.0f, 0.0f, 0.0f));
	texCoords.push_back(vec3(0.0f, 1.0f, 0.0f));
	texCoords.push_back(vec3(1.0f, 1.0f, 0.0f));
	texCoords.push_back(vec3(1.0f, 0.0f, 0.0f));


   uint nrVertices = (uint)vertices.size();
   VertexBufferObject::DATA *attrData = new VertexBufferObject::DATA[nrVertices];

    for(uint i=0; i<nrVertices; ++i)
    {    
        vec3 v = vertices[i];
        vec3 n = normals[i];
		vec3 t = texCoords[i];

        attrData[i].vx = v.x;
        attrData[i].vy = v.y;
        attrData[i].vz = v.z;
        attrData[i].vw = 1.0f;

        attrData[i].nx = n.x;
        attrData[i].ny = n.y;
        attrData[i].nz = n.z;
        attrData[i].nw = 0.0f;

        attrData[i].cx = color.x;
        attrData[i].cy = color.y;
        attrData[i].cz = color.z;
        attrData[i].cw = color.w;

        attrData[i].tx = t.x;
        attrData[i].ty = t.y;
        attrData[i].tz = 0.0f;
		attrData[i].tw = 0.0f;
    }

    VertexBufferObject *vbo = new VertexBufferObject();
    vbo->setData(attrData, GL_STATIC_DRAW, nrVertices, primitive);
    vbo->bindDefaultAttribs();

    delete[] attrData;    

    return vbo;
}

VertexBufferObject *Mesh::box(const vec3 &mi, const vec3 &ma, const vec4 &color, GLenum primitive)
{
    vector<vec3> vertices;
    vector<vec3> normals;

    vertices.push_back(vec3(mi.x, ma.y, mi.z));
    vertices.push_back(vec3(mi.x, ma.y, ma.z));
    vertices.push_back(vec3(ma.x, ma.y, ma.z));
    vertices.push_back(vec3(ma.x, ma.y, mi.z));

    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));


    vertices.push_back(vec3(mi.x, mi.y, mi.z));
    vertices.push_back(vec3(ma.x, mi.y, mi.z));
    vertices.push_back(vec3(ma.x, mi.y, ma.z));
    vertices.push_back(vec3(mi.x, mi.y, ma.z));

    normals.push_back(vec3(0.0f, -1.0f, 0.0f));
    normals.push_back(vec3(0.0f, -1.0f, 0.0f));
    normals.push_back(vec3(0.0f, -1.0f, 0.0f));
    normals.push_back(vec3(0.0f, -1.0f, 0.0f));    


    vertices.push_back(vec3(mi.x, mi.y, mi.z));
    vertices.push_back(vec3(mi.x, ma.y, mi.z));
    vertices.push_back(vec3(ma.x, ma.y, mi.z));
    vertices.push_back(vec3(ma.x, mi.y, mi.z));

    normals.push_back(vec3(0.0f, 0.0f, -1.0f));
    normals.push_back(vec3(0.0f, 0.0f, -1.0f));
    normals.push_back(vec3(0.0f, 0.0f, -1.0f));
    normals.push_back(vec3(0.0f, 0.0f, -1.0f));


    vertices.push_back(vec3(mi.x, mi.y, ma.z));
    vertices.push_back(vec3(ma.x, mi.y, ma.z));
    vertices.push_back(vec3(ma.x, ma.y, ma.z));
    vertices.push_back(vec3(mi.x, ma.y, ma.z));

    normals.push_back(vec3(0.0f, 0.0f, 1.0f));
    normals.push_back(vec3(0.0f, 0.0f, 1.0f));
    normals.push_back(vec3(0.0f, 0.0f, 1.0f));
    normals.push_back(vec3(0.0f, 0.0f, 1.0f));


    vertices.push_back(vec3(mi.x, mi.y, mi.z));
    vertices.push_back(vec3(mi.x, mi.y, ma.z));
    vertices.push_back(vec3(mi.x, ma.y, ma.z));
    vertices.push_back(vec3(mi.x, ma.y, mi.z));

    normals.push_back(vec3(-1.0f, 0.0f, 0.0f));
    normals.push_back(vec3(-1.0f, 0.0f, 0.0f));
    normals.push_back(vec3(-1.0f, 0.0f, 0.0f));
    normals.push_back(vec3(-1.0f, 0.0f, 0.0f));


    vertices.push_back(vec3(ma.x, mi.y, mi.z));
    vertices.push_back(vec3(ma.x, ma.y, mi.z));
    vertices.push_back(vec3(ma.x, ma.y, ma.z));
    vertices.push_back(vec3(ma.x, mi.y, ma.z));

    normals.push_back(vec3(1.0f, 0.0f, 0.0f));
    normals.push_back(vec3(1.0f, 0.0f, 0.0f));
    normals.push_back(vec3(1.0f, 0.0f, 0.0f));
    normals.push_back(vec3(1.0f, 0.0f, 0.0f));


   uint nrVertices = (uint)vertices.size();
   VertexBufferObject::DATA *attrData = new VertexBufferObject::DATA[nrVertices];

    for(uint i=0; i<nrVertices; ++i)
    {    
        vec3 v = vertices[i];
        vec3 n = normals[i];

        attrData[i].vx = v.x;
        attrData[i].vy = v.y;
        attrData[i].vz = v.z;
        attrData[i].vw = 1.0f;

        attrData[i].nx = n.x;
        attrData[i].ny = n.y;
        attrData[i].nz = n.z;
        attrData[i].nw = 0.0f;

        attrData[i].cx = color.x;
        attrData[i].cy = color.y;
        attrData[i].cz = color.z;
        attrData[i].cw = color.w;

        attrData[i].tx = 0.0f;
        attrData[i].ty = 0.0f;
        attrData[i].tz = 0.0f;
        attrData[i].tw = 0.0f;
    }

    VertexBufferObject *vbo = new VertexBufferObject();
    vbo->setData(attrData, GL_STATIC_DRAW, nrVertices, primitive);
    vbo->bindDefaultAttribs();

    delete[] attrData;    

    return vbo;
}

VertexBufferObject *Mesh::sphere(float radius, int iterations, const vec4 &color, GLenum primitive)
{
    vector<FACET3> facets((int)pow(4.0, iterations));
	int n = CreateUnitSphere(facets.data(), iterations);

    vector<vec3> vertices;
    vector<vec3> normals;

    for(int i=0; i<n; ++i)
    {
        FACET3 facet = facets[i];
        vertices.push_back(facet.p1 * radius);
        vertices.push_back(facet.p2 * radius);
        vertices.push_back(facet.p3 * radius);

        normals.push_back(normalize(facet.p1));
        normals.push_back(normalize(facet.p2));
        normals.push_back(normalize(facet.p3));
    }

   uint nrVertices = (uint)vertices.size();
   VertexBufferObject::DATA *attrData = new VertexBufferObject::DATA[nrVertices];

    for(uint i=0; i<nrVertices; ++i)
    {    
        vec3 v = vertices[i];
        vec3 n = normals[i];

        attrData[i].vx = v.x;
        attrData[i].vy = v.y;
        attrData[i].vz = v.z;
        attrData[i].vw = 1.0f;

        attrData[i].nx = n.x;
        attrData[i].ny = n.y;
        attrData[i].nz = n.z;
        attrData[i].nw = 0.0f;

        attrData[i].cx = color.x;
        attrData[i].cy = color.y;
        attrData[i].cz = color.z;
        attrData[i].cw = color.w;

        attrData[i].tx = 0.0f;
        attrData[i].ty = 0.0f;
        attrData[i].tz = 0.0f;
        attrData[i].tw = 0.0f;
    }

    VertexBufferObject *vbo = new VertexBufferObject();
    vbo->setData(attrData, GL_STATIC_DRAW, nrVertices, primitive);
    vbo->bindDefaultAttribs();

    delete[] attrData;    

    return vbo;
}

VertexBufferObject *Mesh::vbo(const vector<vec3> &vertices, const vector<vec4> &colors, const vector<vec3> &normals, GLenum primitive)
{
   uint nrVertices = (uint)vertices.size();
   VertexBufferObject::DATA *attrData = new VertexBufferObject::DATA[nrVertices];

   vec3 n = vec3();
   vec4 c = vec4();

    for(uint i=0; i<nrVertices; ++i)
    {        
        vec3 v = vertices[i];

        if(colors.size() == vertices.size())
        {
            c = colors[i];
        }

        if(normals.size() == vertices.size())
        {
            n = normals[i];
        }
     
        attrData[i].vx = v.x;
        attrData[i].vy = v.y;
        attrData[i].vz = v.z;
        attrData[i].vw = 1.0f;

        attrData[i].nx = n.x;
        attrData[i].ny = n.y;
        attrData[i].nz = n.z;
        attrData[i].nw = 0.0f;

        attrData[i].cx = c.x;
        attrData[i].cy = c.y;
        attrData[i].cz = c.z;
        attrData[i].cw = c.w;

        attrData[i].tx = 0.0f;
        attrData[i].ty = 0.0f;
        attrData[i].tz = 0.0f;
        attrData[i].tw = 0.0f;
    }

    VertexBufferObject *vbo = new VertexBufferObject();
    vbo->setData(attrData, GL_STATIC_DRAW, nrVertices, primitive);
    vbo->bindDefaultAttribs();

    delete[] attrData;    

    return vbo;
}   

VertexBufferObject *vbo(const vector<vec3> &vertices, const vector<vec4> &colors = vector<vec4>(), GLenum primitive = GL_POINTS);

int Mesh::createNSphere(FACET3 *f, int iterations)
{
   int i,it;
   double a;
   vec3 p[6];

   p[0] = vec3(0, 0, 1);
   p[1] = vec3(0, 0, -1);
   p[2] = vec3(-1, -1, 0);
   p[3] = vec3(1, -1, 0);
   p[4] = vec3(1, 1, 0);
   p[5] = vec3(-1, 1, 0);
   
   vec3 pa, pb, pc;
   int nt = 0, ntold;

   /* Create the level 0 object */
   a = 1 / sqrt(2.0);
   for (i=0;i<6;i++) 
   {
      p[i].x *= (float)a;
      p[i].y *= (float)a;
   }

   f[0].p1 = p[0]; f[0].p2 = p[3]; f[0].p3 = p[4];
   f[1].p1 = p[0]; f[1].p2 = p[4]; f[1].p3 = p[5];
   f[2].p1 = p[0]; f[2].p2 = p[5]; f[2].p3 = p[2];
   f[3].p1 = p[0]; f[3].p2 = p[2]; f[3].p3 = p[3];
   f[4].p1 = p[1]; f[4].p2 = p[4]; f[4].p3 = p[3];
   f[5].p1 = p[1]; f[5].p2 = p[5]; f[5].p3 = p[4];
   f[6].p1 = p[1]; f[6].p2 = p[2]; f[6].p3 = p[5];
   f[7].p1 = p[1]; f[7].p2 = p[3]; f[7].p3 = p[2];
   nt = 8;

   if (iterations < 1)
      return(nt);

   /* Bisect each edge and move to the surface of a unit sphere */
   for (it=0;it<iterations;it++) 
   {
      ntold = nt;
      for (i=0;i<ntold;i++) 
      {
         pa.x = (f[i].p1.x + f[i].p2.x) / 2;
         pa.y = (f[i].p1.y + f[i].p2.y) / 2;
         pa.z = (f[i].p1.z + f[i].p2.z) / 2;
         pb.x = (f[i].p2.x + f[i].p3.x) / 2;
         pb.y = (f[i].p2.y + f[i].p3.y) / 2;
         pb.z = (f[i].p2.z + f[i].p3.z) / 2;
         pc.x = (f[i].p3.x + f[i].p1.x) / 2;
         pc.y = (f[i].p3.y + f[i].p1.y) / 2;
         pc.z = (f[i].p3.z + f[i].p1.z) / 2;
         pa.normalize();
         pb.normalize();
         pc.normalize();
         f[nt].p1 = f[i].p1; f[nt].p2 = pa; f[nt].p3 = pc; nt++;
         f[nt].p1 = pa; f[nt].p2 = f[i].p2; f[nt].p3 = pb; nt++;
         f[nt].p1 = pb; f[nt].p2 = f[i].p3; f[nt].p3 = pc; nt++;
         f[i].p1 = pa;
         f[i].p2 = pb;
         f[i].p3 = pc;
      }
   }

   return(nt);
}

int Mesh::CreateUnitSphere(FACET3 *facets, int iterations)
{
	int i,j,n,nstart;
	vec3 p1(1.0,1.0,1.0),  p2(-1.0,-1.0,1.0);
	vec3 p3(1.0,-1.0,-1.0), p4(-1.0,1.0,-1.0); 

	p1.normalize();
	p2.normalize();
	p3.normalize();
	p4.normalize();

    facets[0].p1 = p1; facets[0].p2 = p2; facets[0].p3 = p3;
    facets[1].p1 = p2; facets[1].p2 = p1; facets[1].p3 = p4;
    facets[2].p1 = p2; facets[2].p2 = p4; facets[2].p3 = p3;
    facets[3].p1 = p1; facets[3].p2 = p3; facets[3].p3 = p4;

	n = 4;
	
	for (i=1;i<iterations;i++) 
    {
		nstart = n;

		for (j=0;j<nstart;j++) 
        {
			
			/* Create initially copies for the new facets */
			facets[n  ] = facets[j];
			facets[n+1] = facets[j];
			facets[n+2] = facets[j];

			/* Calculate the midpoints */
			p1 = MidPoint(facets[j].p1,facets[j].p2);
			p2 = MidPoint(facets[j].p2,facets[j].p3);
			p3 = MidPoint(facets[j].p3,facets[j].p1);

			/* Replace the current facet */
			facets[j].p2 = p1;
			facets[j].p3 = p3;

			/* Create the changed vertices in the new facets */
			facets[n  ].p1 = p1;
			facets[n  ].p3 = p2;
			facets[n+1].p1 = p3;
			facets[n+1].p2 = p2;
			facets[n+2].p1 = p1;
			facets[n+2].p2 = p2;
			facets[n+2].p3 = p3;
			n += 3;
		}
	}

	for (j=0;j<n;j++) 
    {
		facets[j].p1.normalize();
		facets[j].p2.normalize();
		facets[j].p3.normalize();
	}

	return(n);
}

vec3 Mesh::MidPoint(vec3 p1, vec3 p2)
{
   vec3 p;

   p.x = (p1.x + p2.x) / 2;
   p.y = (p1.y + p2.y) / 2;
   p.z = (p1.z + p2.z) / 2;

   return(p);
}