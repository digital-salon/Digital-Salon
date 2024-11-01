#include "Mesh2D.h"

Mesh2D::Mesh2D(const int2& nxy, const float2& origin, const float2& end)
{
	Nxy = nxy;
	Origin = origin;
	End = end;

	BuildMesh();
	BuildVbo();
}

Mesh2D::~Mesh2D()
{
}

void Mesh2D::Render()
{
    MeshVbo->render();
    MeshVboWF->render();
}

void Mesh2D::BuildMesh()
{
	// Separation between cells
	float2 ds = (End - Origin);
	ds.x /= 1.f * Nxy.x;
	ds.y /= 1.f * Nxy.y;

	// Mini quads
	Squares = vector<Quad2D>(Nxy.x * Nxy.y);
	for (int i = 0; i < Nxy.x; i++)
	{
		for (int j = 0; j < Nxy.y; j++)
		{
			float3 a = make_float3(Origin.x + (i + 0) * ds.x, Origin.y + (j + 0) * ds.y, 0.f);
			float3 b = make_float3(Origin.x + (i + 1) * ds.x, Origin.y + (j + 0) * ds.y, 0.f);
			float3 c = make_float3(Origin.x + (i + 1) * ds.x, Origin.y + (j + 1) * ds.y, 0.f);
            float3 d = make_float3(Origin.x + (i + 0) * ds.x, Origin.y + (j + 1) * ds.y, 0.f);

			Quad2D tempQuad(a, b, c, d);
			Squares[i + Nxy.x * j] = tempQuad;
		}
	}
}

void Mesh2D::BuildVbo()
{
    vector<float3> positions;
    for (int i = 0; i < Squares.size(); i++)
    {
        Quad2D& t = Squares[i];
        positions.push_back(t.A);
        positions.push_back(t.B);
        positions.push_back(t.C);
        positions.push_back(t.D);
    }

    VertexBufferObject::DATA* data_mesh = new VertexBufferObject::DATA[positions.size()];

    for (uint i = 0; i < positions.size(); i++)
    {
        float3 v = positions[i];

        data_mesh[i].vx = v.x;
        data_mesh[i].vy = v.y;
        data_mesh[i].vz = v.z;
        data_mesh[i].vw = 0.0f;

        data_mesh[i].cx = 1.0f;
        data_mesh[i].cy = 1.0f;
        data_mesh[i].cz = 1.0f;
        data_mesh[i].cw = 1.0f;

        data_mesh[i].nx = 0.0f;
        data_mesh[i].ny = 0.0f;
        data_mesh[i].nz = 0.0f;
        data_mesh[i].nw = 0.0f;

        data_mesh[i].tx = 0.0f;
        data_mesh[i].ty = 0.0f;
        data_mesh[i].tz = 0.0f;
        data_mesh[i].tw = 0.0f;
    }

    MeshVbo = std::make_shared<VertexBufferObject>();
    MeshVbo->setData(data_mesh, GL_DYNAMIC_DRAW, (GLuint)positions.size(), GL_QUADS);
    MeshVbo->bindDefaultAttribs();

    delete[] data_mesh;

    // Grid Lines
    vector<float3> positionsWF;
    for (int i = 0; i < Squares.size(); i++)
    {
        Quad2D& t = Squares[i];
        positionsWF.push_back(t.A);
        positionsWF.push_back(t.B);

        positionsWF.push_back(t.B);
        positionsWF.push_back(t.C);

        positionsWF.push_back(t.C);
        positionsWF.push_back(t.D);

        positionsWF.push_back(t.D);
        positionsWF.push_back(t.A);
    }

    VertexBufferObject::DATA* data_wf = new VertexBufferObject::DATA[positionsWF.size()];

    for (uint i = 0; i < positionsWF.size(); i++)
    {
        float3 v = positionsWF[i];

        data_wf[i].vx = v.x;
        data_wf[i].vy = v.y;
        data_wf[i].vz = v.z;
        data_wf[i].vw = 0.0f;

        data_wf[i].cx = 0.0f;//0.1f;
        data_wf[i].cy = 0.0f;//0.1f;
        data_wf[i].cz = 0.0f;//0.8f;
        data_wf[i].cw = 1.0f;//1.0f;

        data_wf[i].nx = 0.0f;
        data_wf[i].ny = 0.0f;
        data_wf[i].nz = 0.0f;
        data_wf[i].nw = 0.0f;

        data_wf[i].tx = 0.0f;
        data_wf[i].ty = 0.0f;
        data_wf[i].tz = 0.0f;
        data_wf[i].tw = 0.0f;
    }

    MeshVboWF = std::make_shared<VertexBufferObject>();
    MeshVboWF->setData(data_wf, GL_DYNAMIC_DRAW, (GLuint)positionsWF.size(), GL_LINES);
    MeshVboWF->bindDefaultAttribs();

    delete[] data_wf;
}
