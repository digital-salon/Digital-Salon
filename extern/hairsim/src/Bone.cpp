#include "Bone.h"

Bone::Bone()
{
	ID = -1;
	Name = "";
	Rot = Mat4::Identity();
	Move = Mat4::Identity();
	Scale = Mat4::Identity();
}

void Bone::UpdateTrans()
{
	SetMove(MoveVec);
	SetRot(Angles);
	SetScale(make_float3(ScaleNum));
}

Mat4 Bone::GetLocalTrans()
{
	return Move * Rot * Scale;
}

void Bone::SetRot(float3 r)
{
	Rot = Mat4::RotateZ(r.z) * Mat4::RotateY(r.y) * Mat4::RotateX(r.x);
}

void Bone::SetMove(float3 t)
{
	Move = Mat4::Translate(t);
}

void Bone::SetScale(float3 s)
{
	Scale = Mat4::Scale(s);
}
