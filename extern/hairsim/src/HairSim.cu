#include "HairSim.cuh"

__constant__ SimulationParams Params;

void copySimParamsToDevice(SimulationParams* hostParams)
{
	cudaMemcpyToSymbol(Params, hostParams, sizeof(SimulationParams));
}

__host__ __device__ float3 BaryCoordinates(const float3& p, const float3& a, const float3& b, const float3& c)
{
	// Assumes point and triangle coplanar
	float3 v0 = b - a;
	float3 v1 = c - a;
	float3 v2 = p - a;

	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);

	float denom = d00 * d11 - d01 * d01;
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0f - v - w;

	// Clamp to avoid errors
	u = fmin(1.f, fmax(u, 0.f));
	v = fmin(1.f, fmax(v, 0.f));
	w = fmin(1.f, fmax(w, 0.f));

	return make_float3(u, v, w);
}

__device__ float3 ExtForce(const int& globalI, const DeviceBuffers& buffers)
{
	// wind field
	float3 wind = Params.HairMass * Params.WindSpeed;

	// gravity
	float3 gravity = make_float3(0.f, -Params.Gravity * Params.HairMass, 0.f);

	// angular/target spring
	//float3 targetPos = buffers.Particles[globalI - 1].Position + buffers.Particles[globalI].Angular;
	//float3 dir = targetPos - buffers.Particles[globalI].Position;
	//float3 biphasicAng = Params.AngularK * dir;

	// gravity spring
	Particle p = buffers.Particles[globalI];
	float alpha = 1.f;
	float3 edge = p.GravityPos - p.Position;
	float dist = length(edge);
	float3 dir = normalize(edge);

	//return Params.GravityK * alpha * dist * dir;
	float3 grov =  Params.GravityK * alpha * dist * dir;


	return wind + gravity + grov;// +biphasicAng;
}

__device__ float3 SprForce(const int& globalI, const DeviceBuffers& buffers)
{
	// Prepares data
	Particle& p = buffers.Particles[globalI];
	float3 force = make_float3(0.f);
	int idx = p.LocalIdx;

	// left neighbors
	if (idx > 0) force += Params.EdgeK * SprForce(globalI, globalI - 1, p.EdgeRestIdx.x, buffers);
	if (idx > 1) force += Params.BendK * SprForce(globalI, globalI - 2, p.BendRestIdx.x, buffers);
	if (idx > 2) force += Params.TorsionK * SprForce(globalI, globalI - 3, p.TorsRestIdx.x, buffers);

	// right neighbors
	if (idx < p.StrandLength - 1) force += Params.EdgeK * SprForce(globalI, globalI + 1, p.EdgeRestIdx.y, buffers);
	if (idx < p.StrandLength - 2) force += Params.BendK * SprForce(globalI, globalI + 2, p.BendRestIdx.y, buffers);
	if (idx < p.StrandLength - 3) force += Params.TorsionK * SprForce(globalI, globalI + 3, p.TorsRestIdx.y, buffers);

	return force;
}

__device__ float3 SprForce(const int& globalI, const int& globalJ, const int& restIdx, const DeviceBuffers& buffers)
{
	if(buffers.Particles[globalJ].Cut) return make_float3(0.f);
	float3 dir = buffers.Particles[globalJ].Position - buffers.Particles[globalI].Position;
	float l = length(dir);
	float3 dirN = (1.f / l) * dir;
	float cut = (l - buffers.RestLenghts[restIdx]);
	return cut * dirN;
}

__device__ Mat3 DirMat(const int& globalI, const int& globalJ, const DeviceBuffers& buffers)
{
	return Mat3(normalize(buffers.Particles[globalJ].Position - buffers.Particles[globalI].Position));
}

__device__ int3 PosToCellSdf(const float3& p)
{
	// Cell indices (negative or beyond grid size => outside domain)
	int cellX = floor(p.x * Params.SdfInvDs.x);
	int cellY = floor(p.y * Params.SdfInvDs.y);
	int cellZ = floor(p.z * Params.SdfInvDs.z);

	return make_int3(cellX, cellY, cellZ);
}

__device__ int3 PosToCell(const float3& p)
{
	// Cell numbers (negative or beyond grid size => outside domain)
	int cellX = floor(p.x * Params.HairInvDs.x);
	int cellY = floor(p.y * Params.HairInvDs.y);
	int cellZ = floor(p.z * Params.HairInvDs.z);

	return make_int3(cellX, cellY, cellZ);
}

__device__ float3 NormalPos(const float3& p, const int3& cellIdx)
{
	float varX = p.x * Params.HairInvDs.x - cellIdx.x;
	float varY = p.y * Params.HairInvDs.y - cellIdx.y;
	float varZ = p.z * Params.HairInvDs.z - cellIdx.z;

	return make_float3(varX, varY, varZ);
}

__device__ bool CellInsideGrid(const int3& idx)
{
	bool inside = idx.x >= 0 && idx.x < Params.HairDim.x
		&& idx.y >= 0 && idx.y < Params.HairDim.y
		&& idx.z >= 0 && idx.z < Params.HairDim.z;

	return inside;
}

// For trilinear interpolation we need idx to be at most
// on penultime box
__device__ bool CellOutsideInfluence(const int3& idx)
{
	bool inside = idx.x > 0 && idx.x < Params.HairDim.x - 1
		&& idx.y > 0 && idx.y < Params.HairDim.y - 1
		&& idx.z > 0 && idx.z < Params.HairDim.z - 1;

	return !inside;
}

__device__ float DistancePointSegment(const float3& A, const float3& B, const float3& u, float& t)
{
	// Line segment and auxiliary segment
	float3 ab = B - A;
	float3 au = u - A;

	// Projection of au onto ab
	float proj = dot(au, ab);
	float abSqNorm = dot(ab, ab);

	// Clamp value
	t = min(max(0.f, proj / abSqNorm), 1.f);

	// Distance from u to closest point (A + t(B-A))
	return length(u - A - t * ab);
}

// A is stored as: 
// ...a(i,i), a(i,i+1), a(i,i+2), a(i,i+3), a(i,i-1), a(i,i-2), a(i,i-3)...
__inline__ __device__ Mat3 SampleA(const int& globalI, const int& globalJ, const DeviceBuffers& buffers)
{
	int diff = globalJ - globalI;
	return buffers.StrandA[diff >= 0 ? diff : 3 - diff];
}

// L is stored as: ...l(i,i), l(i,i-1), l(i,i-2), l(i,i-3)...
__inline__ __device__ Mat3 SampleL(const int& globalI, const int& globalJ, const DeviceBuffers& buffers)
{
	return buffers.StrandL[4 * globalI + (globalI - globalJ)];
}

// U is stored as: ...u(i,i), u(i,i+1), u(i,i+2), u(i,i+3)...
__inline__ __device__ Mat3 SampleU(const int& globalI, const int& globalJ, const DeviceBuffers& buffers)
{
	return buffers.StrandL[4 * globalI + (globalJ - globalI)];
}

// V is stored as: ...v(i-1), v(i), v(i+1)...
__inline__ __device__ float3 SampleV(const int& globalI, const DeviceBuffers& buffers)
{
	return buffers.StrandV[globalI];
}

// Sample velocity field from averaging staggered grids
// ! this is sampled at center point
__device__ float3 SampleHairPIC(const int3& idx, const DeviceBuffers& buffers)
{
	float picX = 0.5 * (buffers.HairPicU[Idx1DU(idx)] + buffers.HairPicU[Idx1DU(idx + make_int3(1, 0, 0))]);
	float picY = 0.5 * (buffers.HairPicV[Idx1DV(idx)] + buffers.HairPicV[Idx1DV(idx + make_int3(0, 1, 0))]);
	float picZ = 0.5 * (buffers.HairPicW[Idx1DW(idx)] + buffers.HairPicW[Idx1DW(idx + make_int3(0, 0, 1))]);

	return make_float3(picX, picY, picZ);
}

__device__ float3 SampleHairVel(const int3& idx, const DeviceBuffers& buffers)
{
	float velX = 0.5 * (buffers.HairVelU[Idx1D(idx)] + buffers.HairVelU[Idx1D(idx + make_int3(1, 0, 0))]);
	float velY = 0.5 * (buffers.HairVelV[Idx1D(idx)] + buffers.HairVelV[Idx1D(idx + make_int3(0, 1, 0))]);
	float velZ = 0.5 * (buffers.HairVelW[Idx1D(idx)] + buffers.HairVelW[Idx1D(idx + make_int3(0, 0, 1))]);

	return make_float3(velX, velY, velZ);
}

__device__ float3 SampleHairFLIP(const int3& idx, const DeviceBuffers& buffers)
{
	// original rasterized velocity
	float velX = 0.5 * (buffers.HairVelU[Idx1DU(idx)] + buffers.HairVelU[Idx1DU(idx + make_int3(1, 0, 0))]);
	float velY = 0.5 * (buffers.HairVelV[Idx1DV(idx)] + buffers.HairVelV[Idx1DV(idx + make_int3(0, 1, 0))]);
	float velZ = 0.5 * (buffers.HairVelW[Idx1DW(idx)] + buffers.HairVelW[Idx1DW(idx + make_int3(0, 0, 1))]);

	// incompressible solution
	float picX = 0.5 * (buffers.HairPicU[Idx1DU(idx)] + buffers.HairPicU[Idx1DU(idx + make_int3(1, 0, 0))]);
	float picY = 0.5 * (buffers.HairPicV[Idx1DV(idx)] + buffers.HairPicV[Idx1DV(idx + make_int3(0, 1, 0))]);
	float picZ = 0.5 * (buffers.HairPicW[Idx1DW(idx)] + buffers.HairPicW[Idx1DW(idx + make_int3(0, 0, 1))]);

	return make_float3(picX - velX, picY - velY, picZ - velZ);
}

__device__ float SamplePressure(const int3& idx, const DeviceBuffers& buffers)
{
	// Dirichlet zero at bottom
	if (idx.y < 0) return 0.f;

	// Neumann zero at the sides and top
	int nX = max(0, min(idx.x, Params.HairDim.x - 1));
	int nY = max(0, min(idx.y, Params.HairDim.y - 1));
	int nZ = max(0, min(idx.z, Params.HairDim.z - 1));
	int3 nIdx = make_int3(nX, nY, nZ);

	// Atention! We are sampling "old" pressure
	return buffers.HairPressure[Idx1D(nIdx)].x;

}

__device__ float4 TriInterpSdf(const int3& idx, const float3& pos, const DeviceBuffers& buffers)
{
	// Values at surrounding points
	float4 c000 = SampleTotalSdf(idx + make_int3(0, 0, 0), buffers);
	float4 c001 = SampleTotalSdf(idx + make_int3(0, 0, 1), buffers);
	float4 c010 = SampleTotalSdf(idx + make_int3(0, 1, 0), buffers);
	float4 c011 = SampleTotalSdf(idx + make_int3(0, 1, 1), buffers);
	float4 c100 = SampleTotalSdf(idx + make_int3(1, 0, 0), buffers);
	float4 c101 = SampleTotalSdf(idx + make_int3(1, 0, 1), buffers);
	float4 c110 = SampleTotalSdf(idx + make_int3(1, 1, 0), buffers);
	float4 c111 = SampleTotalSdf(idx + make_int3(1, 1, 1), buffers);

	// Normal coordinates (unit cube)
	float3 ud;
	ud.x = Params.SdfInvDs.x * (pos.x - idx.x * Params.SdfDs.x);
	ud.y = Params.SdfInvDs.y * (pos.y - idx.y * Params.SdfDs.y);
	ud.z = Params.SdfInvDs.z * (pos.z - idx.z * Params.SdfDs.z);

	// Normalize the gradient
	float4 res = TriInterpolate<float4>(ud, c000, c001, c010, c011, c100, c101, c110, c111);
	float3 normal = normalize(make_float3(res.x, res.y, res.z));
	res.x = normal.x;
	res.y = normal.y;
	res.z = normal.z;

	res.w -= Params.ThreshSdf;

	return res;
}

__device__ float3 TriInterSdfVel(const int3& idx, const float3& pos, const DeviceBuffers& buffers)
{
	// Values at surrounding points
	float3 c000 = SampleSdfVel(idx + make_int3(0, 0, 0), buffers);
	float3 c001 = SampleSdfVel(idx + make_int3(0, 0, 1), buffers);
	float3 c010 = SampleSdfVel(idx + make_int3(0, 1, 0), buffers);
	float3 c011 = SampleSdfVel(idx + make_int3(0, 1, 1), buffers);
	float3 c100 = SampleSdfVel(idx + make_int3(1, 0, 0), buffers);
	float3 c101 = SampleSdfVel(idx + make_int3(1, 0, 1), buffers);
	float3 c110 = SampleSdfVel(idx + make_int3(1, 1, 0), buffers);
	float3 c111 = SampleSdfVel(idx + make_int3(1, 1, 1), buffers);

	// Normal coordinates (unit cube)
	float3 ud;
	ud.x = Params.SdfInvDs.x * (pos.x - idx.x * Params.SdfDs.x);
	ud.y = Params.SdfInvDs.y * (pos.y - idx.y * Params.SdfDs.y);
	ud.z = Params.SdfInvDs.z * (pos.z - idx.z * Params.SdfDs.z);

	// Trilinear interpolation
	float3 res = TriInterpolate<float3>(ud, c000, c001, c010, c011, c100, c101, c110, c111);

	return res;
}

__device__ float3 SampleNablaSdf(const int3& idx, const DeviceBuffers& buffers)
{
	// We asume zero neumann conditions : d\phi/dn = 0
	if (idx.x < 0 || idx.x>Params.SdfDim.x ||
		idx.y < 0 || idx.y>Params.SdfDim.y ||
		idx.z < 0 || idx.z>Params.SdfDim.z) return make_float3(0.f);

	return buffers.NablaSdf[Idx1DSdf(idx)];
}

__device__ float SampleSdf(const int3& idx, const DeviceBuffers& buffers)
{
	// We asume zero neumann conditions : d\phi/dn = 0 -> \phi_n - \phi_{n-1} = 0
	// -> \phi_n = \phi_{n-1}
	int x = max(0, min(idx.x, Params.SdfDim.x - 1));
	int y = max(0, min(idx.y, Params.SdfDim.y - 1));
	int z = max(0, min(idx.z, Params.SdfDim.z - 1));

	return buffers.Sdf[Idx1DSdf(make_int3(x, y, z))];
}

__device__ float4 SampleTotalSdf(const int3& idx, const DeviceBuffers& buffers)
{
	float3 normal = SampleNablaSdf(idx, buffers);
	float phi = SampleSdf(idx, buffers);
	return make_float4(normal.x, normal.y, normal.z, phi);
}

__device__ float3 SampleSdfVel(const int3& idx, const DeviceBuffers& buffers)
{
	// We asume zero neumann conditions : d\phi/dn = 0 -> \phi_n - \phi_{n-1} = 0
	// -> \phi_n = \phi_{n-1}
	int x = max(0, min(idx.x, Params.SdfDim.x - 1));
	int y = max(0, min(idx.y, Params.SdfDim.y - 1));
	int z = max(0, min(idx.z, Params.SdfDim.z - 1));

	float vx = buffers.HeadVelX[Idx1DSdf(make_int3(x, y, z))];
	float vy = buffers.HeadVelY[Idx1DSdf(make_int3(x, y, z))];
	float vz = buffers.HeadVelZ[Idx1DSdf(make_int3(x, y, z))];

	return make_float3(vx, vy, vz);
}

// Easily get indices of sparse columns in A
__inline__ __device__ int IdxA(const int& globalI, const int& globalJ)
{
	int diff = globalJ - globalI;
	int sparseIdx = diff >= 0 ? diff : 3 - diff;
	return 7 * globalI + sparseIdx;
}

__inline__ __device__ int IdxL(const int& globalI, const int& globalJ)
{
	return 4 * globalI + (globalI - globalJ);
}

__inline__ __device__ int IdxU(const int& globalI, const int& globalJ)
{
	return 4 * globalI + (globalJ - globalI);
}

__inline__ __device__ int3 Idx3D(const int& idx)
{
	int3 result = make_int3(0, 0, 0);
	result.x = idx % Params.HairDim.x;
	result.y = (idx / Params.HairDim.x) % Params.HairDim.y;
	result.z = idx / (Params.HairDim.x * Params.HairDim.y);
	return result;
}

__inline__ __device__ int3 Idx3DSdf(const int& idx)
{
	int3 result = make_int3(0, 0, 0);
	result.x = idx % Params.SdfDim.x;
	result.y = (idx / Params.SdfDim.x) % Params.SdfDim.y;
	result.z = idx / (Params.SdfDim.x * Params.SdfDim.y);
	return result;
}

__inline__ __device__ int3 Idx3DU(const int& idx)
{
	int3 result = make_int3(0, 0, 0);
	result.x = idx % (Params.HairDim.x + 1);
	result.y = (idx / (Params.HairDim.x + 1)) % Params.HairDim.y;
	result.z = idx / ((Params.HairDim.x + 1) * Params.HairDim.y);
	return result;
}

__inline__ __device__ int3 Idx3DV(const int& idx)
{
	int3 result = make_int3(0, 0, 0);
	result.x = idx % Params.HairDim.x;
	result.y = (idx / Params.HairDim.x) % (Params.HairDim.y + 1);
	result.z = idx / (Params.HairDim.x * (Params.HairDim.y + 1));
	return result;
}

__inline__ __device__ int3 Idx3DW(const int& idx)
{
	int3 result = make_int3(0, 0, 0);
	result.x = idx % Params.HairDim.x;
	result.y = (idx / Params.HairDim.x) % Params.HairDim.y;
	result.z = idx / (Params.HairDim.x * Params.HairDim.y);
	return result;
}

__inline__ __device__ int Idx1D(const int3& idx)
{
	return idx.x + idx.y * Params.HairDim.x + idx.z * Params.HairDim.x * Params.HairDim.y;
}

__inline__ __device__ int Idx1DSdf(const int3& idx)
{
	return idx.x + idx.y * Params.SdfDim.x + idx.z * Params.SdfDim.x * Params.SdfDim.y;
}

__inline__ __device__ int Idx1DU(const int3& idx)
{
	return idx.x + idx.y * (Params.HairDim.x + 1) + idx.z * (Params.HairDim.x + 1) * Params.HairDim.y;
}

__inline__ __device__ int Idx1DV(const int3& idx)
{
	return idx.x + idx.y * Params.HairDim.x + idx.z * Params.HairDim.x * (Params.HairDim.y + 1);
}

__inline__ __device__ int Idx1DW(const int3& idx)
{
	return idx.x + idx.y * Params.HairDim.x + idx.z * Params.HairDim.x * Params.HairDim.y;
}

__global__ void ResetParticles(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// Reset dynamics
	p.Position = p.InitialPosition;
	p.Velocity = make_float3(0.f);
}

void launchKernelResetParticles(const int2& blockThread, DeviceBuffers buffers)
{
	ResetParticles << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void InitParticles(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// Initial parameters
	//p.InitialPosition = p.Position;

	// Prepares biphasic interaction

	/* Computes target direction, this is equivalent to
	computing the angle but has been more efficient in 
	tests.*/

	p.GravityPos = p.InitialPosition + (Params.HairMass * Params.Gravity) / Params.GravityK * make_float3(0, 1, 0);
	p.GravityPos0 = p.GravityPos;
}

void launchKernelInitParticles(const int2& blockThread, DeviceBuffers buffers)
{
	InitParticles << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void FillMatrices(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];
	int rootIdx = p.GlobalIdx - p.LocalIdx;

	// Fills matrix entries
	// we avoid re-computing elements for diagonal (check sec. 5.1) 
	Mat3 diag = (1.f + Params.DtN * Params.Mu * Params.Damping) * Mat3::Identity();
	float k[3] = { Params.EdgeK, Params.BendK, Params.TorsionK };

	// iterate (left and right neighbors)
	// boundary condition, root and cut particles produce just a diagonal
	if (p.LocalIdx != 0 && !p.Cut)
	{
		for (int i = max(p.LocalIdx - 3, 0); i <= min(p.StrandLength - 1, p.LocalIdx + 3); i++)
		{
			int absDiff = abs(i - p.LocalIdx);
			if (absDiff > 0)
			{

				if(buffers.Particles[rootIdx + i].Cut)
				{
					buffers.StrandA[IdxA(p.GlobalIdx, rootIdx + i)] = Mat3::Zero();
				}

				else 
				{
					Mat3 offDiag = -pow(Params.DtN, 2.f) * Params.Mu * k[absDiff - 1] * DirMat(p.GlobalIdx, rootIdx + i, buffers);
					buffers.StrandA[IdxA(p.GlobalIdx, rootIdx + i)] = offDiag;
					diag = diag - offDiag;
				}
			}
		}
	}

	// diagonal
	buffers.StrandA[IdxA(p.GlobalIdx, p.GlobalIdx)] = diag;

	// Fills RHS
	float3 b = make_float3(0.f);
	
	// Boundary condition, root does not move, also cut do not move
	if (p.LocalIdx != 0 && !p.Cut)
	{
		b = p.Velocity + Params.DtN * Params.Mu * (ExtForce(p.GlobalIdx, buffers) + SprForce(p.GlobalIdx, buffers));
	}

	buffers.StrandB[p.GlobalIdx] = b;

	//printf("I am %i, velocity is (%f,%f,%f), position (%f,%f,%f), b (%f,%f,%f)\n",
	//	pIdx, p.GlobalIdx, p.Velocity.x, p.Velocity.y, p.Velocity.z,
	//	p.Position.x, p.Position.y, p.Position.z, 
	//	b.x, b.y, b.z);
}

void launchKernelFillMatrices(const int2& blockThread, DeviceBuffers buffers)
{
	FillMatrices << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void SolveVelocity(DeviceBuffers buffers)
{
	INIT_R;
	if (IDX_OUTSIDE_R) return;

	// Parallel over roots/strands
	Particle& p = buffers.Particles[buffers.RootIdx[rIdx]];
	int rootIdx = p.GlobalIdx;

	// LU decomposition for R^{3M X 3M} strand matrix
	// Forward sweep to get L, U, and V'
	for (int i = 0; i < p.StrandLength; i++)
	{

		// lower triangular
		for (int j = max(i - 3, 0); j <= i; j++)
		{
			Mat3 l = buffers.StrandA[IdxA(rootIdx + i, rootIdx + j)];
			for (int k = max(i - 3, 0); k < j; k++)
			{
				l = l - buffers.StrandL[IdxL(rootIdx + i, rootIdx + k)] * buffers.StrandU[IdxU(rootIdx + k, rootIdx + j)];
			}
			buffers.StrandL[IdxL(rootIdx + i, rootIdx + j)] = l;
		}

		// uper triangular
		buffers.StrandU[IdxU(rootIdx + i, rootIdx + i)] = Mat3::Identity();
		Mat3 invL = buffers.StrandL[IdxL(rootIdx + i, rootIdx + i)].Inverse();
		for (int j = i + 1; j <= min(i + 3, p.StrandLength - 1); j++)
		{
			Mat3 u = buffers.StrandA[IdxA(rootIdx + i, rootIdx + j)];
			for (int k = max(j - 3, 0); k < i; k++)
			{
				u = u - buffers.StrandL[IdxL(rootIdx + i, rootIdx + k)] * buffers.StrandU[IdxU(rootIdx + k, rootIdx + j)];
			}
			buffers.StrandU[IdxU(rootIdx + i, rootIdx + j)] = invL * u;
		}

		// intermediate vector
		float3 vp = buffers.StrandB[rootIdx + i];
		for (int j = max(i - 3, 0); j < i; j++)
		{
			vp -= buffers.StrandL[IdxL(rootIdx + i, rootIdx + j)] * buffers.StrandV[rootIdx + j];
		}
		buffers.StrandV[rootIdx + i] = invL * vp;
	}

	// backward sweep to get final V
	for (int i = p.StrandLength - 1; i >= 0; i--)
	{
		float3 v = buffers.StrandV[rootIdx + i];
		for (int j = i + 1; j <= min(i + 3, p.StrandLength - 1); j++)
		{
			v -= buffers.StrandU[IdxU(rootIdx + i, rootIdx + j)] * buffers.Particles[rootIdx + j].Velocity;
		}
		buffers.Particles[rootIdx + i].Velocity = v;
	}


	//// DEBUG
	//// Checks A=LU
	//for (int i = 0; i < p.StrandLength; i++)
	//{
	//	//printf("Row %i of A-LU\n", i);
	//	for (int j = i - 3; j <= i + 3; j++)
	//	{
	//		Mat3 a = buffers.StrandA[IdxA(rootIdx + i, rootIdx + j)];
	//		Mat3 lu = Mat3::Zero();
	//		for (int k = max(0, max(i - 3, j - 3)); k <= min(i, j); k++)
	//		{
	//			lu = lu + buffers.StrandL[IdxL(rootIdx + i, rootIdx + k)] *
	//				buffers.StrandU[IdxU(rootIdx + k, rootIdx + j)];
	//		}
	//		Mat3 diff = a - lu;
	//		if (diff.NormInfty() > 0.0001) printf("bad LU\n");
	//	}
	//}

	//// Checks LVp = b
	//for (int i = 0; i < p.StrandLength; i++)
	//{
	//	//printf("Entry %i of LVp : ", i);
	//	float3 lv = make_float3(0.f);
	//	for (int j = max(0, i - 3); j <= i; j++)
	//	{
	//		lv += buffers.StrandL[IdxL(rootIdx + i, rootIdx + j)] *
	//			buffers.StrandV[rootIdx + j];
	//	}
	//	lv = lv - buffers.StrandB[rootIdx + i];
	//	if (length(lv) > 0.001) printf("Error in LV=b\n");
	//}

	// Checks AV = b
	//||Av-b||_{\infty} < 0.001
	//for (int i = 0; i < p.StrandLength; i++)
	//{
	//	float3 av = make_float3(0.f);
	//	for (int j = max(i - 3, 0); j <= min(i + 3, p.StrandLength - 1); j++)
	//	{
	//		av += buffers.StrandA[IdxA(rootIdx + i, rootIdx + j)] * buffers.Particles[rootIdx + j].Velocity;
	//	}
	//	av = av - buffers.StrandB[rootIdx + i];
	//	if (length(av) > 0.001f) printf("error in Av=b\n");
	//}

}

void launchKernelSolveVelocity(const int2& blockThread, DeviceBuffers buffers)
{
	SolveVelocity << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void PositionUpdate(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// boundary condition (scalp does not move)
	if (p.LocalIdx == 0) return;
	if (p.Cut) return;

	p.Position += Params.DtN * p.Velocity;
}

void launchKernelPositionUpdate(const int2& blockThread, DeviceBuffers buffers)
{
	PositionUpdate << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void SegmentToGrid(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// Last particle does not produce a segment
	if (p.LocalIdx == p.StrandLength - 1) return;

	if(p.Cut || buffers.Particles[pIdx+1].Cut) return;

	// Coordinates from grid system
	float3 posA = p.Position - Params.HairOrig; 
	float3 posB = buffers.Particles[p.GlobalIdx + 1].Position - Params.HairOrig;

	// Segment's middle point
	float3 midPos = 0.5 * (posA + posB);

	// Staggered grids are offset
	float3 posU = midPos - 0.5 * make_float3(0.f, Params.HairDs.y, Params.HairDs.z);
	float3 posV = midPos - 0.5 * make_float3(Params.HairDs.x, 0.f, Params.HairDs.z);
	float3 posW = midPos - 0.5 * make_float3(Params.HairDs.x, Params.HairDs.y, 0.f);
	float3 posGrid[3] = { posU, posV, posW };

	// U,V,W - grids
	for (int grid = 1; grid <= 3; grid++)
	{
		// Grid size
		int sizeX = grid == 1 ? Params.HairDim.x + 1 : Params.HairDim.x;
		int sizeY = grid == 2 ? Params.HairDim.y + 1 : Params.HairDim.y;
		int sizeZ = grid == 3 ? Params.HairDim.z + 1 : Params.HairDim.z;

		// Closest (floor) cell
		int3 cellIdx = PosToCell(posGrid[grid - 1]);
		bool inside = cellIdx.x >= 0 && cellIdx.x < sizeX
			&& cellIdx.y >= 0 && cellIdx.y < sizeY
			&& cellIdx.z >= 0 && cellIdx.z < sizeZ;

		// Outside grid => do nothing
		if (inside)
		{
			// Checks within a given boundary
			for (int i = max(0, cellIdx.x - Params.NumGridNeighbors); i <= min(sizeX - 1, cellIdx.x + Params.NumGridNeighbors); i++)
			{
				for (int j = max(0, cellIdx.y - Params.NumGridNeighbors); j <= min(sizeY - 1, cellIdx.y + Params.NumGridNeighbors); j++)
				{
					for (int k = max(0, cellIdx.z - Params.NumGridNeighbors); k <= min(sizeZ - 1, cellIdx.z + Params.NumGridNeighbors); k++)
					{
						// Node position (from grid system)
						float3 cellPos = make_float3(Params.HairDs.x * i, Params.HairDs.y * j, Params.HairDs.z * k);
						int3 nIdx = make_int3(i, j, k);
						float t;

						// Computes weight and interpolated velocity
						float weight = fmax(0.f, Params.MaxWeight - DistancePointSegment(posA, posB, cellPos, t));
						float3 vel = (1.f - t) * p.Velocity + t * buffers.Particles[p.GlobalIdx + 1].Velocity;

						// Different grids
						if (grid == 1)
						{
							atomicAdd(&buffers.HairVelU[Idx1DU(nIdx)], weight * vel.x);
							atomicAdd(&buffers.HairWeightU[Idx1DU(nIdx)], weight);
						}
						if (grid == 2)
						{
							atomicAdd(&buffers.HairVelV[Idx1DV(nIdx)], weight * vel.y);
							atomicAdd(&buffers.HairWeightV[Idx1DV(nIdx)], weight);
						}
						if (grid == 3)
						{
							atomicAdd(&buffers.HairVelW[Idx1DW(nIdx)], weight * vel.z);
							atomicAdd(&buffers.HairWeightW[Idx1DW(nIdx)], weight);
						}
					}
				}
			}
		}
	}
}

__global__ void ResetGridU(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GU) return;

	buffers.HairWeightU[cellIdx] = 0.f;
	buffers.HairVelU[cellIdx] = 0.f;
	buffers.HairPicU[cellIdx] = 0.f;
}

__global__ void ResetGridV(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GV) return;

	buffers.HairWeightV[cellIdx] = 0.f;
	buffers.HairVelV[cellIdx] = 0.f;
	buffers.HairPicV[cellIdx] = 0.f;
}

__global__ void ResetGridW(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GW) return;

	buffers.HairWeightW[cellIdx] = 0.f;
	buffers.HairVelW[cellIdx] = 0.f;
	buffers.HairPicW[cellIdx] = 0.f;
}

__global__ void NormalizeGridU(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GU) return;

	if (buffers.HairWeightU[cellIdx] > 0.f)
	{
		buffers.HairVelU[cellIdx] = buffers.HairVelU[cellIdx] / buffers.HairWeightU[cellIdx];
	}
}

__global__ void NormalizeGridV(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GV) return;

	if (buffers.HairWeightV[cellIdx] > 0.f)
	{
		buffers.HairVelV[cellIdx] = buffers.HairVelV[cellIdx] / buffers.HairWeightV[cellIdx];
	}
}

__global__ void NormalizeGridW(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GW) return;

	if (buffers.HairWeightW[cellIdx] > 0.f)
	{
		buffers.HairVelW[cellIdx] = buffers.HairVelW[cellIdx] / buffers.HairWeightW[cellIdx];
	}
}

__global__ void FillVoxelType(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_G) return;

	// Computes total hair-weight at the center of the voxel
	int3 idx = Idx3D(cellIdx);
	bool inside =
		buffers.HairWeightU[Idx1DU(idx + make_int3(1, 0, 0))] > 0.f &&
		buffers.HairWeightV[Idx1DV(idx + make_int3(0, 1, 0))] > 0.f &&
		buffers.HairWeightW[Idx1DW(idx + make_int3(0, 0, 1))] > 0.f &&
		buffers.HairWeightU[Idx1DU(idx)] > 0.f &&
		buffers.HairWeightV[Idx1DV(idx)] > 0.f &&
		buffers.HairWeightW[Idx1DW(idx)] > 0.f;

	// Distinguish air and fluid
	buffers.VoxelType[cellIdx] = inside ? FLUID : AIR;
}

void launchKernelSegmentToGrid(const int2& blockThreadParticles, const int2& blockThreadGrid, const int2& blockThreadU, const int2& blockThreadV, const int2& blockThreadW, DeviceBuffers buffers)
{

	// First, set all grids again to zero
	ResetGridU << <blockThreadU.x, blockThreadU.y >> > (buffers);
	ResetGridV << <blockThreadV.x, blockThreadV.y >> > (buffers);
	ResetGridW << <blockThreadW.x, blockThreadW.y >> > (buffers);

	// Then, each segment adds (using atomics) its contribution
	SegmentToGrid << <blockThreadParticles.x, blockThreadParticles.y >> > (buffers);

	// Normalize all grids
	NormalizeGridU << <blockThreadU.x, blockThreadU.y >> > (buffers);
	NormalizeGridV << <blockThreadV.x, blockThreadV.y >> > (buffers);
	NormalizeGridW << <blockThreadW.x, blockThreadW.y >> > (buffers);

	// Categorize every cell
	FillVoxelType << <blockThreadGrid.x, blockThreadGrid.y >> > (buffers);

}

__global__ void GridToParticle(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// Boundary condition (root does not move)
	if (p.LocalIdx == 0) return;
	if (p.Cut) return;

	// Position from grid's pov (center of voxels)
	float3 pos = p.Position - Params.HairOrig - 0.5 * Params.HairDs;
	int3 cellIdx = PosToCell(pos);

	// Outside of voxel space || outside of fluid => do nothing
	if (CellOutsideInfluence(cellIdx)) return;
	if (buffers.VoxelType[Idx1D(cellIdx)] != FLUID) return;

	// Manual tri-interpolation of velocity (to avoid air cells)
	float3 ud = NormalPos(pos, cellIdx);

	float w000 = buffers.VoxelType[Idx1D(cellIdx + make_int3(0, 0, 0))] == FLUID ? (1.f - ud.x) * (1.f - ud.y) * (1.f - ud.z) : 0.f;
	float w001 = buffers.VoxelType[Idx1D(cellIdx + make_int3(0, 0, 1))] == FLUID ? (1.f - ud.x) * (1.f - ud.y) * (0.f + ud.z) : 0.f;
	float w010 = buffers.VoxelType[Idx1D(cellIdx + make_int3(0, 1, 0))] == FLUID ? (1.f - ud.x) * (0.f + ud.y) * (1.f - ud.z) : 0.f;
	float w011 = buffers.VoxelType[Idx1D(cellIdx + make_int3(0, 1, 1))] == FLUID ? (1.f - ud.x) * (0.f + ud.y) * (0.f + ud.z) : 0.f;
	float w100 = buffers.VoxelType[Idx1D(cellIdx + make_int3(1, 0, 0))] == FLUID ? (0.f + ud.x) * (1.f - ud.y) * (1.f - ud.z) : 0.f;
	float w101 = buffers.VoxelType[Idx1D(cellIdx + make_int3(1, 0, 1))] == FLUID ? (0.f + ud.x) * (1.f - ud.y) * (0.f + ud.z) : 0.f;
	float w110 = buffers.VoxelType[Idx1D(cellIdx + make_int3(1, 1, 0))] == FLUID ? (0.f + ud.x) * (0.f + ud.y) * (1.f - ud.z) : 0.f;
	float w111 = buffers.VoxelType[Idx1D(cellIdx + make_int3(1, 1, 1))] == FLUID ? (0.f + ud.x) * (0.f + ud.y) * (0.f + ud.z) : 0.f;
	float wTotal = w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111;

	// Sample velocity voxel center
	float3 v000 = SampleHairPIC(cellIdx + make_int3(0, 0, 0), buffers);
	float3 v001 = SampleHairPIC(cellIdx + make_int3(0, 0, 1), buffers);
	float3 v010 = SampleHairPIC(cellIdx + make_int3(0, 1, 0), buffers);
	float3 v011 = SampleHairPIC(cellIdx + make_int3(0, 1, 1), buffers);
	float3 v100 = SampleHairPIC(cellIdx + make_int3(1, 0, 0), buffers);
	float3 v101 = SampleHairPIC(cellIdx + make_int3(1, 0, 1), buffers);
	float3 v110 = SampleHairPIC(cellIdx + make_int3(1, 1, 0), buffers);
	float3 v111 = SampleHairPIC(cellIdx + make_int3(1, 1, 1), buffers);
	float3 vPicTotal = w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011
		+ w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111;

	v000 = SampleHairFLIP(cellIdx + make_int3(0, 0, 0), buffers);
	v001 = SampleHairFLIP(cellIdx + make_int3(0, 0, 1), buffers);
	v010 = SampleHairFLIP(cellIdx + make_int3(0, 1, 0), buffers);
	v011 = SampleHairFLIP(cellIdx + make_int3(0, 1, 1), buffers);
	v100 = SampleHairFLIP(cellIdx + make_int3(1, 0, 0), buffers);
	v101 = SampleHairFLIP(cellIdx + make_int3(1, 0, 1), buffers);
	v110 = SampleHairFLIP(cellIdx + make_int3(1, 1, 0), buffers);
	v111 = SampleHairFLIP(cellIdx + make_int3(1, 1, 1), buffers);
	float3 vFlipTotal = w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011
		+ w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111;
		
	float3 velPic = (1.f / wTotal) * vPicTotal;
	float3 velFlip = (1.f / wTotal) * vFlipTotal;

	// Update velocity
	p.Velocity = Params.FlipWeight * (p.Velocity + velFlip) + (1.f - Params.FlipWeight) * velPic;
}

void launchKernelGridToParticle(const int2& blockThread, DeviceBuffers buffers)
{
	GridToParticle << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void ComputeDivergence(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_G) return;

	// Init air pressure/div to zero
	if (buffers.VoxelType[cellIdx] == AIR)
	{
		buffers.HairPressure[cellIdx] = make_float2(0.f);
		buffers.HairDiv[cellIdx] = 0.f;
	}

	// Process only FLUID cells
	if (buffers.VoxelType[cellIdx] != FLUID) return;

	// Compute using centered differences
	int3 idx = Idx3D(cellIdx);
	buffers.HairDiv[cellIdx] =
		(buffers.HairVelU[Idx1DU(idx + make_int3(1, 0, 0))] - buffers.HairVelU[Idx1DU(idx)]) * Params.HairInvDs.x +
		(buffers.HairVelV[Idx1DV(idx + make_int3(0, 1, 0))] - buffers.HairVelV[Idx1DV(idx)]) * Params.HairInvDs.y +
		(buffers.HairVelW[Idx1DW(idx + make_int3(0, 0, 1))] - buffers.HairVelW[Idx1DW(idx)]) * Params.HairInvDs.z;
}

__global__ void JacobiIteration(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_G) return;

	// Boundary condition, air stays at p=0
	if (buffers.VoxelType[cellIdx] == AIR) return;

	int3 idx = Idx3D(cellIdx);
	float invSq = 1.f / (2.f * (Params.HairInvSqDs.x + Params.HairInvSqDs.y + Params.HairInvSqDs.z));
	float newP = invSq * (-buffers.HairDiv[cellIdx]
		+ Params.HairInvSqDs.x * (SamplePressure(idx + make_int3(1, 0, 0), buffers) + SamplePressure(idx + make_int3(-1, 0, 0), buffers))
		+ Params.HairInvSqDs.y * (SamplePressure(idx + make_int3(0, 1, 0), buffers) + SamplePressure(idx + make_int3(0, -1, 0), buffers))
		+ Params.HairInvSqDs.z * (SamplePressure(idx + make_int3(0, 0, 1), buffers) + SamplePressure(idx + make_int3(0, 0, -1), buffers)));

	// Update using weighted Jacobi
	buffers.HairPressure[cellIdx].y = (1.f - Params.JacobiWeight) * buffers.HairPressure[cellIdx].x + Params.JacobiWeight * newP;

}

__global__ void SwapPressure(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_G) return;

	// 'x' holds the "old" pressure
	buffers.HairPressure[cellIdx].x = buffers.HairPressure[cellIdx].y;
}

__global__ void ProjectVelocityU(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GU) return;

	// Corresponding 3D index
	int3 idx = Idx3DU(cellIdx);

	// boundary condition 
	// Neumann zero for pressure at walls => velocity does not change
	if (idx.x == 0 || idx.x == Params.HairDim.x)
	{
		buffers.HairPicU[cellIdx] = buffers.HairVelU[cellIdx];
		return;
	}

	// Otherwise, process only if there is adjacent fluid
	if (buffers.VoxelType[Idx1D(idx)] != FLUID && buffers.VoxelType[Idx1D(idx + make_int3(-1, 0, 0))] != FLUID) return;

	// substract pressure gradient
	float dp = buffers.HairPressure[Idx1D(idx)].x - buffers.HairPressure[Idx1D(idx + make_int3(-1, 0, 0))].x;
	buffers.HairPicU[cellIdx] = buffers.HairVelU[cellIdx] - Params.HairInvDs.x * dp;
}

__global__ void ProjectVelocityV(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GV) return;

	// Corresponding 3D index
	int3 idx = Idx3DV(cellIdx);

	// boundary condition 
	// Neumann zero for pressure at top => velocity does not change
	if (idx.y == Params.HairDim.y)
	{
		buffers.HairPicV[cellIdx] = buffers.HairVelV[cellIdx];
		return;
	}
	// Dirichlet zero for pressure at bot => dp = p
	if (idx.y == 0)
	{
		float dp = buffers.HairPressure[Idx1D(idx)].x;
		buffers.HairPicV[cellIdx] = buffers.HairVelV[cellIdx] - Params.HairInvDs.y * dp;
		return;
	}

	// Otherwise process only if there is adjacent fluid
	if (buffers.VoxelType[Idx1D(idx)] != FLUID && buffers.VoxelType[Idx1D(idx + make_int3(0, -1, 0))] != FLUID) return;

	// substract pressure gradient
	float dp = buffers.HairPressure[Idx1D(idx)].x - buffers.HairPressure[Idx1D(idx + make_int3(0, -1, 0))].x;
	buffers.HairPicV[cellIdx] = buffers.HairVelV[cellIdx] - Params.HairInvDs.y * dp;
}

__global__ void ProjectVelocityW(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_GW) return;

	// Corresponding 3D index
	int3 idx = Idx3DW(cellIdx);

	// boundary condition 
	// Neumann zero for pressure at walls => velocity does not change
	if (idx.z == 0 || idx.z == Params.HairDim.z)
	{
		buffers.HairPicW[cellIdx] = buffers.HairVelW[cellIdx];
		return;
	}

	// Otherwise, process only if there is adjacent fluid
	if (buffers.VoxelType[Idx1D(idx)] != FLUID && buffers.VoxelType[Idx1D(idx + make_int3(0, 0, -1))] != FLUID) return;

	// substract pressure gradient
	float dp = buffers.HairPressure[Idx1D(idx)].x - buffers.HairPressure[Idx1D(idx + make_int3(0, 0, -1))].x;
	buffers.HairPicW[cellIdx] = buffers.HairVelW[cellIdx] - Params.HairInvDs.z * dp;
}

__global__ void DebugPressureSolver(DeviceBuffers buffers, int opt)
{
	INIT_G;
	if (IDX_OUTSIDE_G) return;

	// Process only FLUID cells
	if (buffers.VoxelType[cellIdx] != FLUID) return;
	//if (cellIdx != 36254) return;
	int3 idx = Idx3D(cellIdx);

	float3 ds = Params.HairDs;
	float* velU = buffers.HairVelU;
	float* velV = buffers.HairVelV;
	float* velW = buffers.HairVelW;
	float* picU = buffers.HairPicU;
	float* picV = buffers.HairPicV;
	float* picW = buffers.HairPicW;
	float2* p = buffers.HairPressure;
	int currX = Idx1DU(idx);
	int currY = Idx1DV(idx);
	int currZ = Idx1DW(idx);
	int nextX = Idx1DU(idx + make_int3(1, 0, 0));
	int nextY = Idx1DV(idx + make_int3(0, 1, 0));
	int nextZ = Idx1DW(idx + make_int3(0, 0, 1));
	int prevX = Idx1DU(idx + make_int3(-1, 0, 0));
	int prevY = Idx1DV(idx + make_int3(0, -1, 0));
	int prevZ = Idx1DW(idx + make_int3(0, 0, -1));

	if (opt == 0)
	{
		// Check \nabla^{2} p = \nabla \cdot v
		float invSq = 1.f / (2.f * (Params.HairInvSqDs.x + Params.HairInvSqDs.y + Params.HairInvSqDs.z));
		float error = -SamplePressure(idx, buffers) + invSq * (-buffers.HairDiv[cellIdx]
			+ Params.HairInvSqDs.x * (SamplePressure(idx + make_int3(1, 0, 0), buffers) + SamplePressure(idx + make_int3(-1, 0, 0), buffers))
			+ Params.HairInvSqDs.y * (SamplePressure(idx + make_int3(0, 1, 0), buffers) + SamplePressure(idx + make_int3(0, -1, 0), buffers))
			+ Params.HairInvSqDs.z * (SamplePressure(idx + make_int3(0, 0, 1), buffers) + SamplePressure(idx + make_int3(0, 0, -1), buffers)));
		if (abs(error) > 0.0001)printf("cell %i pressure solver error %f\n", cellIdx, abs(error));
	}

	//Check \nabla \cdot v_{pic} = 0 
	if (opt == 1)
	{
		float div =
			(buffers.HairPicU[Idx1DU(idx + make_int3(1, 0, 0))] - buffers.HairPicU[Idx1DU(idx)]) * Params.HairInvDs.x +
			(buffers.HairPicV[Idx1DV(idx + make_int3(0, 1, 0))] - buffers.HairPicV[Idx1DV(idx)]) * Params.HairInvDs.y +
			(buffers.HairPicW[Idx1DW(idx + make_int3(0, 0, 1))] - buffers.HairPicW[Idx1DW(idx)]) * Params.HairInvDs.z;

		float3 picP = make_float3(picU[currX], picV[currY], picW[currZ]);
		float3 picN = make_float3(picU[nextX], picV[nextY], picW[nextZ]);
		float3 velP = make_float3(velU[currX], velV[currY], velW[currZ]);
		float3 velN = make_float3(velU[nextX], velV[nextY], velW[nextZ]);
		float3 preP = make_float3(p[prevX].x, p[prevY].x, p[prevZ].x);
		float3 preN = make_float3(p[nextX].x, p[nextY].x, p[nextZ].x);
		//printf("ds (%f,%f,%f\n", ds.x, ds.y, ds.z);
		//printf("pic(% f, % f, % f) picN(% f, % f, % f)\n",
		//	picP.x, picP.y, picP.z,
		//	picN.x, picN.y, picN.z);
		//printf("cell 3D (%i,%i,%i) vel (%f,%f,%f) velN (%f,%f,%f) pleft (%f,%f,%f) pRight (%f,%f,%f)\n", idx.x, idx.y, idx.z,
		//	velP.x, velP.y, velP.z, 
		//	velN.x, velN.y, velN.z,
		//	preP.x, preP.y, preP.z,
		//	preN.x, preN.y, preN.z,
		//	buffers.HairPressure[cellIdx].x);
		if (abs(div) > 0.0001) printf("cell %i div new %f div old %f pressure %f\n", cellIdx, div, buffers.HairDiv[cellIdx], buffers.HairPressure[cellIdx].y);
	}
}

void launchKernelProjectVelocity(const int2& blockThreadGrid, const int2& blockThreadU, const int2& blockThreadV, const int2& blockThreadW, const int& numIter, DeviceBuffers buffers)
{
	// First, inits pressure everywhere and get fluid divergence
	ComputeDivergence << <blockThreadGrid.x, blockThreadGrid.y >> > (buffers);

	// Solves the pressure equation \nabla^{2} p = \nabla \cdot v
	for (int i = 0; i < numIter; i++)
	{
		JacobiIteration << <blockThreadGrid.x, blockThreadGrid.y >> > (buffers);
		SwapPressure << <blockThreadGrid.x, blockThreadGrid.y >> > (buffers);
	}

	// Projects velocity field
	ProjectVelocityU << <blockThreadU.x, blockThreadU.y >> > (buffers);
	ProjectVelocityV << <blockThreadV.x, blockThreadV.y >> > (buffers);
	ProjectVelocityW << <blockThreadW.x, blockThreadW.y >> > (buffers);

	// Debug whole projection step
	//DebugPressureSolver << <blockThreadGrid.x, blockThreadGrid.y >> > (buffers, 1);
}

__global__ void HeadCollision(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// Boundary condition (scalp does not move)
	if (p.LocalIdx == 0 || p.Cut) return;

	// Check particle's index within the box
	//float3 pos = p.Position - Params.HairOrig;
	float3 posPre = p.Position - Params.HairOrig;
	float3 pos;
	pos.x = dot(posPre, Params.HairAxis[0]);
	pos.y = dot(posPre, Params.HairAxis[1]);
	pos.z = dot(posPre, Params.HairAxis[2]);
	int3 cellIdx = PosToCellSdf(pos);

	// Outside box, do nothing
	bool inside = cellIdx.x > 0 && cellIdx.x < Params.SdfDim.x - 1
		&& cellIdx.y > 0 && cellIdx.y < Params.SdfDim.y - 1
		&& cellIdx.z > 0 && cellIdx.z < Params.SdfDim.z - 1;

	if (!inside) return;

	// Otherwise, gets (via trilinear interp.) SDF and normal vector
	float4 sdf = TriInterpSdf(cellIdx, pos, buffers);

	// Positive SDF -> Do nothing
	if (sdf.w >= 0) return;

	// 2 : Mesh vel., particle vel., normal, and sdf
	float3 vMesh = TriInterSdfVel(cellIdx, pos, buffers);
	float3 vPart = p.Velocity;
	float3 normal = make_float3(sdf.x, sdf.y, sdf.z);
	float phi = sdf.w;

	// 3 : Decompose into normal and tangential
	float3 vMesh_n = dot(vMesh, normal) * normal;
	float3 vMesh_t = vMesh - vMesh_n;
	float3 vPart_n = dot(vPart, normal) * normal;
	float3 vPart_t = vPart - vPart_n;

	// Vel correction
	float rel = length(vPart_n - vMesh_n) / length(vPart_t - vMesh_t);
	float3 velNew = vMesh + fmax(0.f, 1 - Params.Friction * rel) * (vPart_t - vMesh_t);
	//float3 velNew = vMesh + Params.Friction * (vPart_t - vMesh_t);

	// 5: Updates
	p.Velocity = velNew;
	p.Position = p.Position0 + Params.Dt * p.Velocity;

	// Second stage

	// Check particle's index within the box
	posPre = p.Position - Params.HairOrig;
	pos;
	pos.x = dot(posPre, Params.HairAxis[0]);
	pos.y = dot(posPre, Params.HairAxis[1]);
	pos.z = dot(posPre, Params.HairAxis[2]);
	cellIdx = PosToCellSdf(pos);

	// Outside box, do nothing
	inside = cellIdx.x > 0 && cellIdx.x < Params.SdfDim.x - 1
		&& cellIdx.y > 0 && cellIdx.y < Params.SdfDim.y - 1
		&& cellIdx.z > 0 && cellIdx.z < Params.SdfDim.z - 1;

	if (inside)
	{
		// Otherwise, gets (via trilinear interp.) SDF and normal vector
		sdf = TriInterpSdf(cellIdx, pos, buffers);

		// Positive SDF -> Do nothing
		if (sdf.w < 0)
		{
			p.Position -= phi * normal;
			return;
		}
	}

}

void launchKernelHeadCollision(const int2& blockThread, DeviceBuffers buffers)
{
	HeadCollision << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void NablaSdf(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_SDF) return;

	// 3D index
	int3 idx3D = Idx3DSdf(cellIdx);

	// Samples SDF
	float dXb = SampleSdf(idx3D + make_int3(1, 0, 0), buffers);
	float dXa = SampleSdf(idx3D + make_int3(-1, 0, 0), buffers);
	float dYb = SampleSdf(idx3D + make_int3(0, 1, 0), buffers);
	float dYa = SampleSdf(idx3D + make_int3(0, -1, 0), buffers);
	float dZb = SampleSdf(idx3D + make_int3(0, 0, 1), buffers);
	float dZa = SampleSdf(idx3D + make_int3(0, 0, -1), buffers);

	// Second-order finite difference derivative
	float dS_dx = 0.5 * Params.SdfInvDs.x * (dXb - dXa);
	float dS_dy = 0.5 * Params.SdfInvDs.y * (dYb - dYa);
	float dS_dz = 0.5 * Params.SdfInvDs.z * (dZb - dZa);

	// Writes on buffer
	buffers.NablaSdf[cellIdx] = make_float3(dS_dx, dS_dy, dS_dz);
}

void launchKernelNablaSdf(const int2& blockThread, DeviceBuffers buffers)
{
	NablaSdf << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void InitHeadVel(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_SDF) return;

	buffers.HeadVelX[cellIdx] = 0.f;
	buffers.HeadVelY[cellIdx] = 0.f;
	buffers.HeadVelZ[cellIdx] = 0.f;
}

void launchKernelInitHeadVel(const int2& blockThread, DeviceBuffers buffers)
{
	InitHeadVel << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void InitHeadVerticesVel(DeviceBuffers buffers)
{
	INIT_V;
	if (IDX_OUTSIDE_VERT) return;

	buffers.HeadVertices[vIdx].Vel = make_float3(0.f);
	buffers.HeadVertices[vIdx].PosPrevAnim = buffers.HeadVertices[vIdx].Pos;
}

void launchKernelInitHeadVerticesVel(const int2& blockThread, DeviceBuffers buffers)
{
	InitHeadVerticesVel << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void ResetSdfVelocity(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_SDF) return;

	buffers.HeadVelX[cellIdx] = 0.f;
	buffers.HeadVelY[cellIdx] = 0.f;
	buffers.HeadVelZ[cellIdx] = 0.f;
}

__global__ void SdfVelToGrid(DeviceBuffers buffers)
{
	INIT_V;
	if (IDX_OUTSIDE_VERT) return;

	SimpleVertex& v = buffers.HeadVertices[vIdx];

	// Coordinates from grid system
	float3 pos = v.Pos - Params.HairOrig;

	// just one (centered) grid
	int3 cellIdx = PosToCellSdf(pos);
	bool inside = cellIdx.x >= 0 && cellIdx.x < Params.SdfDim.x - 1
		&& cellIdx.y >= 0 && cellIdx.y < Params.SdfDim.y - 1
		&& cellIdx.z >= 0 && cellIdx.z < Params.SdfDim.z - 1;

	// Outside grid -> do nothing
	if (inside)
	{
		int numNeigh = 1;
		float maxDs = fmax(Params.SdfDs.x, fmax(Params.SdfDs.y, Params.SdfDs.z));
		float maxWeight = sqrtf(3) * numNeigh * maxDs;

		// Affects neighbors
		for (int i = max(0, cellIdx.x - numNeigh); i <= min(Params.SdfDim.x - 1, cellIdx.x + numNeigh); i++)
		{
			for (int j = max(0, cellIdx.y - numNeigh); j <= min(Params.SdfDim.y - 1, cellIdx.y + numNeigh); j++)
			{
				for (int k = max(0, cellIdx.z - numNeigh); k <= min(Params.SdfDim.z - 1, cellIdx.z + numNeigh); k++)
				{
					// Node position (from grid system)
					float3 cellPos = make_float3(Params.SdfDs.x * i, Params.SdfDs.y * j, Params.SdfDs.z * k);
					int3 nIdx = make_int3(i, j, k);

					// weight is point distance
					float weight = maxWeight - length(pos - cellPos);
					float3 vel = v.Vel;

					atomicAdd(&buffers.HeadVelX[Idx1DSdf(nIdx)], weight * vel.x);
					atomicAdd(&buffers.HeadVelY[Idx1DSdf(nIdx)], weight * vel.y);
					atomicAdd(&buffers.HeadVelZ[Idx1DSdf(nIdx)], weight * vel.z);

					atomicAdd(&buffers.HeadVelWeight[Idx1DSdf(nIdx)], weight);
				}
			}
		}
	}
}

__global__ void SdfVelNormalize(DeviceBuffers buffers)
{
	INIT_G;
	if (IDX_OUTSIDE_SDF) return;

	if (buffers.HeadVelWeight[cellIdx] > 0.f)
	{
		buffers.HeadVelX[cellIdx] = buffers.HeadVelX[cellIdx] / buffers.HeadVelWeight[cellIdx];
		buffers.HeadVelY[cellIdx] = buffers.HeadVelY[cellIdx] / buffers.HeadVelWeight[cellIdx];
		buffers.HeadVelZ[cellIdx] = buffers.HeadVelZ[cellIdx] / buffers.HeadVelWeight[cellIdx];
	}
}

void launchKernelUpdateVelocitySdf(const int2& blockThreadSdf, const int2& blockThreadVertices, DeviceBuffers buffers)
{

	// First, cleans velocity field
	ResetSdfVelocity << <blockThreadSdf.x, blockThreadSdf.y >> > (buffers);

	// Rastrer to grid
	SdfVelToGrid << <blockThreadVertices.x, blockThreadVertices.y >> > (buffers);

	// Normalize
	SdfVelNormalize << <blockThreadSdf.x, blockThreadSdf.y >> > (buffers);
}

__global__ void StrainLimiting(DeviceBuffers buffers)
{
	INIT_R;
	if (IDX_OUTSIDE_R) return;

	// Parallel over roots/strands
	Particle& p = buffers.Particles[buffers.RootIdx[rIdx]];
	int rootIdx = p.GlobalIdx;

	// Marches from the root
	for (int i = 0; i < p.StrandLength - 1; i++)
	{

		if(buffers.Particles[rootIdx+i].Cut || buffers.Particles[rootIdx+i+1].Cut) 
		{

		}
		else
		{
			// Geometric info
			Particle& pCurrent = buffers.Particles[rootIdx + i];
			Particle& pNext = buffers.Particles[rootIdx + i + 1];
			float& l0 = buffers.RestLenghts[pCurrent.EdgeRestIdx.y];
			float l = length(pNext.Position - pCurrent.Position);

			// Distance constraint
			if (abs(l - l0) > Params.StrainError * l0)
			{
				float3 dir = normalize(pCurrent.Position - pNext.Position);
				pNext.Position += (l - l0) * dir;
		}
		}
	}
}

void launchKernelStrainLimiting(const int2& blockThread, DeviceBuffers buffers)
{
	StrainLimiting << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void SwapPositions(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];
	p.Position0 = p.Position;
}

void launchKernelSwapPositions(const int2& blockThread, DeviceBuffers buffers)
{	
	SwapPositions << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void MoveRoots(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];


	// Root
	if(p.LocalIdx==0) p.Position = buffers.RigidMotion[0] * p.InitialPosition;

	if(p.LocalIdx!=0) p.GravityPos = buffers.RigidMotion[0] * p.GravityPos0;

	//if (p.InitialPosition.x > 0.f) return;

	// Manual translation for debbuging
	//p.Position = p.InitialPosition + Params.RootMove;

}

void launchKernelMoveRoots(const int2& blockThread, DeviceBuffers buffers)
{
	MoveRoots << <blockThread.x, blockThread.y >> > (buffers);
}

__global__ void UpdateMesh(DeviceBuffers buffers)
{
	INIT_T;
	if(IDX_OUTSIDE_TRIANGLE) return;

	SimpleTriangle& t = buffers.HeadTriangles[tIdx];
}

void launchKernelUpdateMesh(const int2 &blockThread, DeviceBuffers buffers)
{
	UpdateMesh<<<blockThread.x, blockThread.y>>>(buffers);
}

__global__ void CutHair(DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	if(p.Cut) p.Position = buffers.Particles[p.CutParent].Position;
}

void launchKernelCutHair(const int2 &blockThread, DeviceBuffers buffers)
{
	CutHair <<<blockThread.x, blockThread.y>>> (buffers);
}

__device__ bool InsideCylinder(float3 pos, float3 dir, float3 q)
{
	// vector from pos to Q
	float3 w = q - pos;

	// projection of w onto cylinder axis
	float3 w_proj = (dot(w, dir) / dot(dir, dir)) * dir;

	// perpendicular component
	float3 w_otrho = w - w_proj;

	// distance to axis
	float d = length(w_otrho);

	return d > Params.CutRadius;
}

__global__ void CutSelect(float3 u, float3 dir, DeviceBuffers buffers)
{

	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];

	// Boundary condition (scalp does not move)
	if (p.LocalIdx == 0) return;
	if (p.LocalIdx <= Params.CutMin) return;

	// Checks if cylinder equation is satisfied
	float3 pos = p.Position;
	if(powf(length(pos - u), 2) <= Params.CutRadius + powf(dot(pos - u, dir), 2))
	{
		p.Cut = true;
	}
}

__global__ void CutCheck(DeviceBuffers buffers)
{
	INIT_R;
	if(IDX_OUTSIDE_R) return;

	// Iterate over strands i.e. roots
	int rootIdx = buffers.RootIdx[rIdx];

	Particle& root = buffers.Particles[rootIdx];

	// Find first cut particle
	int idxParent = rootIdx + root.StrandLength;
	for(int i = 0; i < root.StrandLength; i++)
	{
		Particle& p = buffers.Particles[rootIdx + i];
		if(p.Cut) 
		{
			idxParent = rootIdx + i - 1;
			break;
		}
	}

	// Makes sure everything after cut parent is cut
	for(int i = idxParent + 1; i < rootIdx + root.StrandLength; i++)
	{
		buffers.Particles[i].Cut = true;
		buffers.Particles[i].CutParent = idxParent;
	}


}

void launchKernelCutSelect(const int2 &blockthread, const int2 &blockthreadRoot, float3 pos, float3 dir, DeviceBuffers buffers)
{
	CutSelect<<<blockthread.x, blockthread.y>>> (pos, dir, buffers);
	CutCheck<<<blockthreadRoot.x, blockthreadRoot.y>>> (buffers);
}

__global__ void UpdateRootsSeq(DeviceBuffers buffers)
{
	INIT_R;
	if (IDX_OUTSIDE_R) return;

	Particle& p = buffers.Particles[buffers.RootIdx[rIdx]];

	// Gets attachment triangle
	SimpleTriangle& tri = buffers.HeadTriangles[buffers.RootTri[rIdx]];

	// Current positions
	SimpleVertex& vA = buffers.HeadVertices[tri.V[0]];
	SimpleVertex& vB = buffers.HeadVertices[tri.V[1]];
	SimpleVertex& vC = buffers.HeadVertices[tri.V[2]];

	// Move root following barycentric coordinates
	float3& bary = buffers.RootBary[rIdx];
	p.Position = bary.x * vA.Pos + bary.y * vB.Pos + bary.z * vC.Pos;

}

void launchKernelUpdateRootsSeq(const int2& blockthread, DeviceBuffers buffers)
{
	UpdateRootsSeq << <blockthread.x, blockthread.y >> > (buffers);
}

__global__ void UpdateAnimSeq(int a, int b, float lambda, DeviceBuffers buffers)
{
	INIT_V;
	if (IDX_OUTSIDE_VERT) return;

	// Unwraps data
	MiniVertex& vA = buffers.AnimVertices[a * Params.NumVertices + vIdx];
	MiniVertex& vB = buffers.AnimVertices[b * Params.NumVertices + vIdx];
	SimpleVertex& v = buffers.HeadVertices[vIdx];

	// Linearly interpolates between frames
	v.Pos = (1.f - lambda) * vA.Pos + lambda * vB.Pos;
	v.Normal = (1.f - lambda) * vA.Normal + lambda * vB.Normal;

	// updates velocity
	v.Vel = Params.InvDt * (v.Pos - v.PosPrevAnim);
	v.PosPrevAnim = v.Pos;

}

void launchKernelUpdateAnimSeq(const int2& blockthread, const int& a, const int& b, const float& lambda, DeviceBuffers buffers)
{
	UpdateAnimSeq << <blockthread.x, blockthread.y >> > (a, b, lambda, buffers);
}

__global__ void CudaToGLMesh(float* meshVbo, DeviceBuffers buffers)
{
	INIT_T;
	if (IDX_OUTSIDE_TRIANGLE) return;

	SimpleTriangle& t = buffers.HeadTriangles[tIdx];
	Mat4& trans = buffers.RigidMotion[0];
	Mat4& inv = buffers.RigidMotion[1];

	// Positions and normals
	for (int i = 0; i < 3; i++)
	{
		SimpleVertex& v = buffers.HeadVertices[t.V[i]];
		float3 pos = trans * v.Pos;
		float3 normal = inv * v.Normal;

		meshVbo[20 * (3 * t.Idx + i) + 0] = pos.x;
		meshVbo[20 * (3 * t.Idx + i) + 1] = pos.y;
		meshVbo[20 * (3 * t.Idx + i) + 2] = pos.z;

		meshVbo[20 * (3 * t.Idx + i) + 4] = normal.x;
		meshVbo[20 * (3 * t.Idx + i) + 5] = normal.y;
		meshVbo[20 * (3 * t.Idx + i) + 6] = normal.z;
	}
}

void launchKernelCudaToGLMesh(const int2& blockThread, float* meshVbo, DeviceBuffers buffers)
{
	CudaToGLMesh << <blockThread.x, blockThread.y >> > (meshVbo, buffers);
}

__global__ void CudaToGLInter(float* hairVbo, DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_INTER) return;

	InterParticle& p = buffers.InterParticles[pIdx];
	int2& idx = p.VboIdx;
	float3 posA = buffers.Particles[p.InterIdx.x].Position;
	float3 posB = buffers.Particles[p.InterIdx.y].Position;


	float3 pos = p.Lambda * posB + (1.f-p.Lambda) * posA;

	if (idx.x >= 0)
	{
		hairVbo[20 * idx.x + 0] = pos.x;
		hairVbo[20 * idx.x + 1] = pos.y;
		hairVbo[20 * idx.x + 2] = pos.z;
	}

	if (idx.y >= 0)
	{
		hairVbo[20 * idx.y + 0] = pos.x;
		hairVbo[20 * idx.y + 1] = pos.y;
		hairVbo[20 * idx.y + 2] = pos.z;
	}
}


void launchKernelCudaToGLInter(const int2 &blockThread, float *hairVbo, DeviceBuffers buffers)
{
	CudaToGLInter<<<blockThread.x, blockThread.y>>> (hairVbo, buffers);
}

__global__ void CudaToGLHair(float* hairVbo, DeviceBuffers buffers)
{
	INIT_P;
	if (IDX_OUTSIDE_P) return;

	Particle& p = buffers.Particles[pIdx];
	int2& idx = p.VboIdx;

	float3 pos = p.Position;

	if (idx.x >= 0)
	{
		hairVbo[20 * idx.x + 0] = pos.x;
		hairVbo[20 * idx.x + 1] = pos.y;
		hairVbo[20 * idx.x + 2] = pos.z;
	}

	if (idx.y >= 0)
	{
		hairVbo[20 * idx.y + 0] = pos.x;
		hairVbo[20 * idx.y + 1] = pos.y;
		hairVbo[20 * idx.y + 2] = pos.z;
	}
}

void launchKernelCudaToGLHair(const int2 &blockThread, float *hairVbo, DeviceBuffers buffers)
{
	CudaToGLHair << <blockThread.x, blockThread.y >> > (hairVbo, buffers);
}

__global__ void SetRandom(curandState* state)
{
	INIT_G;
	if (IDX_OUTSIDE_G) return;

	curand_init(cellIdx * cellIdx, cellIdx, 0, &state[cellIdx]);
}

void launchKernelSetRandom(const int2& blockThread, curandState* state)
{
	SetRandom << <blockThread.x, blockThread.y >> > (state);
}
