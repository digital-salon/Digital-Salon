#include "VdbExp.h"

#ifdef WITH_OPENVDB

VdbExp::VdbExp()
{
    openvdb::initialize();
}

void VdbExp::AddGridData(const std::string& name, float* data, int3 dim, float scale, bool sdf)
{
    // Create a shared pointer to a newly-allocated grid of a built-in type:
    // in this case, a FloatGrid, which stores one single-precision floating point
    // value per voxel.  Other built-in grid types include BoolGrid, DoubleGrid,
    // Int32Grid and Vec3SGrid (see openvdb.h for the complete list).
    // The grid comprises a sparse tree representation of voxel data,
    // user-supplied metadata and a voxel space to world space transform,
    // which defaults to the identity transform.
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();

    // Name the grid
    grid->setName(name);

    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (int x = 0; x < dim.x; x++)
    {
        for (int y = 0; y < dim.y; y++)
        {
            for (int z = 0; z < dim.z; z++)
            {
                // swizzle y and z
                float value = scale * data[ArrayIdx(x, y, z, dim)];
                if (sdf)
                {
                    if (value > 0) value = 0.f;
                    else value = -1.f * value;
                }
                accessor.setValue(openvdb::Coord(x, z, y), value);
            }
        }
    }

    Grids.push_back(grid);
}


void VdbExp::Save(const std::string name, const bool& add_timestamp)
{
    std::stringstream ss;
    ss << name;
    // add timestamp
    if (add_timestamp) ss << "_" << std::chrono::system_clock::now().time_since_epoch().count();
    // add suffix
    ss << ".vdb";

    // Create a VDB file object.
    openvdb::io::File file(ss.str());

    // Write out the contents of the container.
    file.write(Grids);
    file.close();
}


int VdbExp::ArrayIdx(const int& x, const int& y, const int& z, const int3& dim)
{
    return (z * dim.y + y) * dim.x + x;
}

#endif