// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD
//  This file is part of the CubeRigIK UE 5.4 plugin. It CANNOT be compiled in
//  the cube-bauhaus repository (no Unreal Engine headers are present here). It
//  is written to be correct and idiomatic for UE 5.4 — drop the plugin into a
//  UE 5.4 project and build it there. See integration/ue5/README.md.
// =============================================================================
//
// Shared, header-only marshalling helpers between the cube-rig C ABI
// (cube_rig.h) and Unreal's math/container types. Factored out so the Blueprint
// library, the UActorComponent, and the AnimGraph node all pack bones the same
// way (the column-major Mat4 convention the C ABI documents).
//
// IMPORTANT — single coordinate space:
//   The solver is engine-agnostic and performs NO handedness/unit conversion.
//   Every array you feed a single solve call (local_bind, target, obstacle
//   capsules) MUST be expressed in ONE consistent space. Throughout this plugin
//   that space is the SkeletalMeshComponent's *component space* in UE world
//   units (centimeters, left-handed, Z-up). Pick component space (not world)
//   because bone transforms are naturally component-relative and the result we
//   write back is per-bone local rotation, which is space-independent once the
//   binds + target agree.

#pragma once

#include "CoreMinimal.h"

// The hand-authored C ABI from cube-rig-ffi. PublicIncludePaths in Build.cs adds
// ThirdParty/CubeRigFFI/include so this resolves in a real UE build.
THIRD_PARTY_INCLUDES_START
#include "cube_rig.h"
THIRD_PARTY_INCLUDES_END

namespace CubeRigIKMarshal
{
	/**
	 * Pack a UE FMatrix into the column-major 16-float layout glam's
	 * Mat4::from_cols_array expects (the layout the C ABI documents).
	 *
	 * UE's FMatrix stores rows in M[row][col] and uses row-vector math
	 * (v' = v * M). glam/cube-rig is column-major with column-vector math
	 * (v' = M * v). The numerical transpose of the storage converts between the
	 * two: cols[c*4 + r] = M[c][r]. Because FMatrix is stored as M[row][col]
	 * row-major, writing M[c][r] for c=column, r=row yields the column-major
	 * cols-array directly.
	 *
	 * This is byte-for-byte the same convention as
	 * UCubeRigIKBlueprintLibrary::PackColumnMajor (kept in one place here).
	 */
	FORCEINLINE void PackColumnMajor(const FMatrix& M, float Out[16])
	{
		for (int32 Col = 0; Col < 4; ++Col)
		{
			for (int32 Row = 0; Row < 4; ++Row)
			{
				Out[Col * 4 + Row] = static_cast<float>(M.M[Col][Row]);
			}
		}
	}

	/** Convenience: pack an FTransform's matrix (with scale) column-major. */
	FORCEINLINE void PackTransform(const FTransform& T, float Out[16])
	{
		PackColumnMajor(T.ToMatrixWithScale(), Out);
	}

	/** Pack an FVector into a float[3] (single-precision, solver-side). */
	FORCEINLINE void PackVector(const FVector& V, float Out[3])
	{
		Out[0] = static_cast<float>(V.X);
		Out[1] = static_cast<float>(V.Y);
		Out[2] = static_cast<float>(V.Z);
	}

	/** Build a CrfCapsule from two endpoints (already in the solver space). */
	FORCEINLINE CrfCapsule MakeCapsule(const FVector& A, const FVector& B, float Radius)
	{
		CrfCapsule C;
		PackVector(A, C.a);
		PackVector(B, C.b);
		C.radius = Radius;
		return C;
	}

	/** Unmarshal one solved (x,y,z,w) entry from the flat output buffer. */
	FORCEINLINE FQuat UnpackQuat(const float* Out, int32 BoneIndex)
	{
		return FQuat(
			Out[BoneIndex * 4 + 0],
			Out[BoneIndex * 4 + 1],
			Out[BoneIndex * 4 + 2],
			Out[BoneIndex * 4 + 3]);
	}
}
