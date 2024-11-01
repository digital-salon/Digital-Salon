#pragma once

#ifdef WITH_OPENVDB

#include <vector>
#define OPENVDB_OPENEXR_STATICLIB
#include <openvdb/openvdb.h>

#include <helper_math.h>

using namespace std;

class VdbExp
{
public:
	VdbExp();
	void AddGridData(const string& name, float* data, int3 dim, float scale, bool sdf);
	void Save(const string name = "myGrid", const bool& add_timestamp = false);
	int ArrayIdx(const int& x, const int& y, const int& z, const int3& dim);

private:
	openvdb::GridPtrVec Grids;
};

#endif