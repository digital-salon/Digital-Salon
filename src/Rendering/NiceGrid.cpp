#include "NiceGrid.h"
#include "Shader.h"
#include "VertexBufferObject.h"
#include "Texture.h"
#include "Mesh.h"

NiceGrid::NiceGrid(GLfloat size, GLfloat rep)
: m_size(size),
  m_rep(rep),
  m_backFaceAlpha(0.2f),
  m_material(vec3(0.3f), vec3(1.0f), vec3(0.2f), 10.0f, "../data/floor_grey.png"), 
  m_vbo(nullptr),
  m_position(0.0f, -45.0f, 0.0f)
{
    buildVBO();
}

NiceGrid::~NiceGrid()
{
    delete m_vbo;
}

void NiceGrid::render(const Transform &trans)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    mat4 model = mat4::translate(m_position);

    glEnable(GL_TEXTURE_2D);

    Shader *shader = shaders::inst()->grid;
    shader->bind();  

        shader->set3f("camPos", params::inst()->camPos);
        shader->setf("alpha", 1.0);
        shader->seti("applyShadow", params::inst()->applyShadow);
        shader->seti("renderMode", params::inst()->gridRenderMode);
        shader->set3f("ambient", vec3(0.2f));
        shader->setf("shadowIntensity", params::inst()->shadowIntensity);
        
        shader->setLights(params::inst()->lights);
        shader->setMatrices(trans, model, false, false, false, true, false, true);
        shader->setMaterial(m_material);

        glEnable(GL_CULL_FACE);    
        glCullFace(GL_BACK);

        m_vbo->render();

        glCullFace(GL_FRONT);
        shader->setf("alpha", m_backFaceAlpha);

        m_vbo->render();
    shader->release();

    glPopClientAttrib();
    glPopAttrib();
}

void NiceGrid::buildVBO()
{
    float s = m_size;
    float r = m_rep;

    vector<Vertex> vertices;
    vertices.push_back(Vertex(vec3(-s, 0.0f, -s), vec3(0, 1, 0), vec4(1), vec2(0, 0)));
    vertices.push_back(Vertex(vec3(-s, 0.0f,  s), vec3(0, 1, 0), vec4(1), vec2(0, r)));
    vertices.push_back(Vertex(vec3( s, 0.0f,  s), vec3(0, 1, 0), vec4(1), vec2(r, r)));
    vertices.push_back(Vertex(vec3( s, 0.0f, -s), vec3(0, 1, 0), vec4(1), vec2(r, 0)));

    VertexBufferObject::DATA *data = new VertexBufferObject::DATA[vertices.size()];

	for (uint i = 0; i<vertices.size(); ++i)
	{
		vec3 v = vertices[i].position;
		vec3 n = vertices[i].normal;
		vec2 t = vertices[i].texture;
        vec4 c = vertices[i].color;

		data[i].vx = v.x;
		data[i].vy = v.y;
		data[i].vz = v.z;
		data[i].vw = 0.0f;

        data[i].cx = c.x;
        data[i].cy = c.y;
        data[i].cz = c.z;
		data[i].cw = c.w;
			
        data[i].nx = n.x;
		data[i].ny = n.y;
		data[i].nz = n.z;
		data[i].nw = 0.0f;
            
		data[i].tx = t.x;
		data[i].ty = t.y;
        data[i].tz = 0.0f;
        data[i].tw = 0.0f;
	}

    m_vbo = new VertexBufferObject();
    m_vbo->setData(data, GL_STATIC_DRAW, (GLuint)vertices.size(), GL_QUADS);
    m_vbo->bindDefaultAttribs();

    delete[] data;
}