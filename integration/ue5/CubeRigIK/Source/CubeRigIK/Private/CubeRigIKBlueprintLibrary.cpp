// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
#include "CubeRigIKBlueprintLibrary.h"

#include "Logging/LogMacros.h"

// The hand-authored C ABI from cube-rig-ffi. PublicIncludePaths in Build.cs adds
// ThirdParty/CubeRigFFI/include so this resolves.
THIRD_PARTY_INCLUDES_START
#include "cube_rig.h"
THIRD_PARTY_INCLUDES_END

DEFINE_LOG_CATEGORY_STATIC(LogCubeRigIK, Log, All);

namespace
{
	/**
	 * Pack a UE FMatrix into the column-major 16-float layout glam's
	 * Mat4::from_cols_array expects (the layout the C ABI documents).
	 *
	 * UE's FMatrix stores rows in M[row][col] and uses row-vector math
	 * (v' = v * M). glam/cube-rig is column-major with column-vector math
	 * (v' = M * v). The numerical transpose of the storage converts between the
	 * two: element (col, row) of the cols-array is FMatrix.M[col][row] laid out
	 * so that consecutive 4 floats form one column.
	 *
	 * Concretely: cols[c*4 + r] = M[c][r]. Because FMatrix is stored as
	 * M[row][col] row-major, writing M[c][r] for c=column,r=row yields the
	 * column-major array directly.
	 */
	void PackColumnMajor(const FMatrix& M, float Out[16])
	{
		for (int32 Col = 0; Col < 4; ++Col)
		{
			for (int32 Row = 0; Row < 4; ++Row)
			{
				// cols-array column `Col`, entry `Row`.
				Out[Col * 4 + Row] = static_cast<float>(M.M[Col][Row]);
			}
		}
	}
}

bool UCubeRigIKBlueprintLibrary::SolveArmIK(
	const TArray<FTransform>& BoneBinds,
	const TArray<int32>& Parents,
	const TArray<int32>& Chain,
	int32 Effector,
	FVector Target,
	const FString& PriorJsonPath,
	int32 Iterations,
	TArray<FQuat>& OutRotations)
{
	OutRotations.Reset();

	const int32 BoneCount = BoneBinds.Num();
	if (BoneCount == 0 || Parents.Num() != BoneCount || Chain.Num() == 0)
	{
		UE_LOG(LogCubeRigIK, Warning,
			TEXT("SolveArmIK: invalid input (BoneCount=%d, Parents=%d, Chain=%d)"),
			BoneCount, Parents.Num(), Chain.Num());
		return false;
	}

	// ── Marshal: parents (int32, -1 => root) -> uint64 (CRF_NO_PARENT) ───────
	TArray<uint64> CParents;
	CParents.SetNumUninitialized(BoneCount);
	for (int32 i = 0; i < BoneCount; ++i)
	{
		CParents[i] = (Parents[i] < 0)
			? CRF_NO_PARENT
			: static_cast<uint64>(Parents[i]);
	}

	// ── Marshal: bind transforms -> column-major Mat4 floats ─────────────────
	TArray<float> CLocalBind;
	CLocalBind.SetNumUninitialized(BoneCount * 16);
	for (int32 i = 0; i < BoneCount; ++i)
	{
		PackColumnMajor(BoneBinds[i].ToMatrixWithScale(), &CLocalBind[i * 16]);
	}

	// ── Marshal: chain (int32 -> uint64) ─────────────────────────────────────
	TArray<uint64> CChain;
	CChain.SetNumUninitialized(Chain.Num());
	for (int32 i = 0; i < Chain.Num(); ++i)
	{
		CChain[i] = static_cast<uint64>(FMath::Max(0, Chain[i]));
	}

	const float CTarget[3] = {
		static_cast<float>(Target.X),
		static_cast<float>(Target.Y),
		static_cast<float>(Target.Z),
	};

	// ── Optional prior: load from JSON path (handle freed before return) ─────
	CrfPrior* Prior = nullptr;
	if (!PriorJsonPath.IsEmpty())
	{
		Prior = crf_prior_load_json(TCHAR_TO_UTF8(*PriorJsonPath));
		if (Prior == nullptr)
		{
			UE_LOG(LogCubeRigIK, Warning,
				TEXT("SolveArmIK: failed to load prior '%s'; using uniform."),
				*PriorJsonPath);
		}
	}

	// ── Output buffer (xyzw per bone) ────────────────────────────────────────
	TArray<float> COut;
	COut.SetNumZeroed(BoneCount * 4);

	const int Code = crf_solve_chain(
		CParents.GetData(),
		CLocalBind.GetData(),
		static_cast<size_t>(BoneCount),
		CChain.GetData(),
		static_cast<size_t>(CChain.Num()),
		static_cast<uint64>(FMath::Max(0, Effector)),
		CTarget,
		Prior,
		nullptr,                 // v1: no hinge limits
		FMath::Max(1, Iterations),
		COut.GetData());

	if (Prior != nullptr)
	{
		crf_prior_free(Prior);
	}

	if (Code != CRF_OK)
	{
		UE_LOG(LogCubeRigIK, Warning, TEXT("SolveArmIK: solver returned error %d"), Code);
		return false;
	}

	// ── Unmarshal: xyzw floats -> FQuat ──────────────────────────────────────
	OutRotations.SetNumUninitialized(BoneCount);
	for (int32 i = 0; i < BoneCount; ++i)
	{
		OutRotations[i] = FQuat(
			COut[i * 4 + 0],
			COut[i * 4 + 1],
			COut[i * 4 + 2],
			COut[i * 4 + 3]);
	}
	return true;
}

FString UCubeRigIKBlueprintLibrary::GetCubeRigVersion()
{
	const char* V = crf_version();
	return V ? FString(UTF8_TO_TCHAR(V)) : FString();
}
