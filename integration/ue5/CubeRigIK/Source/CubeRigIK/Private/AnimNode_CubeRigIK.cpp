// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================

#include "AnimNode_CubeRigIK.h"

#include "CubeRigIKMarshal.h"

#include "Animation/AnimInstanceProxy.h"
#include "AnimationRuntime.h"
#include "Logging/LogMacros.h"

DEFINE_LOG_CATEGORY_STATIC(LogCubeRigIKNode, Log, All);

FAnimNode_CubeRigIK::~FAnimNode_CubeRigIK()
{
	if (Prior != nullptr)
	{
		crf_prior_free(Prior);
		Prior = nullptr;
	}
}

void FAnimNode_CubeRigIK::EnsurePriorLoaded()
{
	if (bPriorLoadAttempted)
	{
		return;
	}
	bPriorLoadAttempted = true;
	if (!PriorJsonPath.IsEmpty())
	{
		Prior = crf_prior_load_json(TCHAR_TO_UTF8(*PriorJsonPath));
		if (Prior == nullptr)
		{
			UE_LOG(LogCubeRigIKNode, Warning, TEXT("CubeRigIK AnimNode: failed to load prior '%s'; using uniform."), *PriorJsonPath);
		}
	}
}

bool FAnimNode_CubeRigIK::IsValidToEvaluate(const USkeleton* Skeleton, const FBoneContainer& RequiredBones)
{
	if (!EffectorBone.IsValidToEvaluate(RequiredBones))
	{
		return false;
	}
	for (const FBoneReference& B : ChainBones)
	{
		if (!B.IsValidToEvaluate(RequiredBones))
		{
			return false;
		}
	}
	return ChainBones.Num() > 0;
}

void FAnimNode_CubeRigIK::InitializeBoneReferences(const FBoneContainer& RequiredBones)
{
	EffectorBone.Initialize(RequiredBones);
	for (FBoneReference& B : ChainBones)
	{
		B.Initialize(RequiredBones);
	}
	// Defer prior loading; it is path-only and worker-safe once loaded.
}

void FAnimNode_CubeRigIK::EvaluateSkeletalControl_AnyThread(
	FComponentSpacePoseContext& Output,
	TArray<FBoneTransform>& OutBoneTransforms)
{
	// THREAD: worker thread. Read-only on UObjects; only the pose + POD inputs.
	check(OutBoneTransforms.Num() == 0);

	EnsurePriorLoaded(); // path-only; safe here (see header).

	const FBoneContainer& BoneContainer = Output.Pose.GetPose().GetBoneContainer();
	const FReferenceSkeleton& RefSkel = BoneContainer.GetReferenceSkeleton();
	const int32 BoneCount = RefSkel.GetNum();
	if (BoneCount == 0)
	{
		return;
	}

	// =========================================================================
	// 1. HARVEST  -- parents[], local_bind[] from the REFERENCE skeleton.
	// =========================================================================
	// We harvest the full-skeleton reference (bind) locals: stable, topologically
	// ordered (UE guarantees parent < child), and exactly the `local_bind` the C
	// ABI wants. The live pose is used below only to place the goal/space, not to
	// re-derive binds — keeping the bind pose fixed makes the solve deterministic.
	TArray<uint64> CParents;
	CParents.SetNumUninitialized(BoneCount);
	TArray<float> CLocalBind;
	CLocalBind.SetNumUninitialized(BoneCount * 16);

	const TArray<FTransform>& RefPose = RefSkel.GetRefBonePose();
	for (int32 i = 0; i < BoneCount; ++i)
	{
		const int32 Parent = RefSkel.GetParentIndex(i);
		CParents[i] = (Parent == INDEX_NONE) ? CRF_NO_PARENT : static_cast<uint64>(Parent);
		CubeRigIKMarshal::PackTransform(RefPose[i], &CLocalBind[i * 16]);
	}

	// Chain + effector as SKELETON bone indices (the C ABI indexes the full
	// skeleton arrays above). FBoneReference caches a compact index; map it back
	// to the skeleton index via the bone container.
	auto SkeletonIndexOf = [&](const FBoneReference& Ref) -> int32
	{
		const FCompactPoseBoneIndex Compact = Ref.GetCompactPoseIndex(BoneContainer);
		if (Compact == INDEX_NONE)
		{
			return INDEX_NONE;
		}
		// GetSkeletonIndex returns the index into the SAME reference skeleton we
		// read above (BoneContainer.GetReferenceSkeleton()), so it indexes our
		// CLocalBind/CParents arrays directly.
		return BoneContainer.GetSkeletonIndex(Compact);
	};

	TArray<uint64> CChain;
	CChain.Reserve(ChainBones.Num());
	for (const FBoneReference& B : ChainBones)
	{
		const int32 Idx = SkeletonIndexOf(B);
		if (Idx == INDEX_NONE || Idx >= BoneCount)
		{
			return; // bone not in this LOD/skeleton; skip the solve gracefully.
		}
		CChain.Add(static_cast<uint64>(Idx));
	}
	const int32 EffectorSkelIdx = SkeletonIndexOf(EffectorBone);
	if (EffectorSkelIdx == INDEX_NONE || EffectorSkelIdx >= BoneCount)
	{
		return;
	}

	// =========================================================================
	// 2. GOAL  -- already in component space (the pin contract).
	// =========================================================================
	float CTarget[3];
	CubeRigIKMarshal::PackVector(EffectorTargetComponent, CTarget);

	// =========================================================================
	// 3. OBSTACLES  -- supplied upstream in component space (may be empty).
	// =========================================================================
	const CrfCapsule* ObstaclePtr = Obstacles.Num() > 0 ? Obstacles.GetData() : nullptr;

	// =========================================================================
	// 4. SOLVE
	// =========================================================================
	TArray<float> COut;
	COut.SetNumZeroed(BoneCount * 4);
	const int Code = crf_solve_chain_avoid(
		CParents.GetData(),
		CLocalBind.GetData(),
		static_cast<size_t>(BoneCount),
		CChain.GetData(),
		static_cast<size_t>(CChain.Num()),
		static_cast<uint64>(EffectorSkelIdx),
		CTarget,
		Prior,
		nullptr,
		FMath::Max(1, Iterations),
		ObstaclePtr,
		static_cast<size_t>(Obstacles.Num()),
		BoneRadius,
		FMath::Max(1, AvoidIterations),
		COut.GetData());

	if (Code != CRF_OK)
	{
		UE_LOG(LogCubeRigIKNode, Verbose, TEXT("CubeRigIK AnimNode: solver returned %d"), Code);
		return; // leave pose unchanged.
	}

	// =========================================================================
	// 5. APPLY  -- write solved CHAIN locals as component-space FBoneTransforms.
	// =========================================================================
	// The solver returns LOCAL rotations. To emit FBoneTransforms (which the
	// skeletal-control base expects in COMPONENT space), we walk the chain
	// parent->child, composing each solved local rotation onto the running
	// component transform. We keep each bone's existing component translation
	// (from its parent's solved component transform * the bind local translation)
	// so bone lengths are preserved.
	//
	// OutBoneTransforms must be sorted by ascending compact index; we add chain
	// bones (topologically ordered) then sort to satisfy the contract.

	auto SolvedLocal = [&](int32 SkelIdx) -> FTransform
	{
		// Solved rotation + the bind local translation/scale.
		FTransform T = RefPose[SkelIdx];
		T.SetRotation(CubeRigIKMarshal::UnpackQuat(COut.GetData(), SkelIdx));
		return T;
	};

	// Build component-space transforms for each chain bone by composing from the
	// chain root's parent's CURRENT component-space transform.
	for (int32 ci = 0; ci < CChain.Num(); ++ci)
	{
		const int32 SkelIdx = static_cast<int32>(CChain[ci]);
		// Skeleton index -> compact pose index (the inverse of GetSkeletonIndex).
		const FCompactPoseBoneIndex CompactIdx =
			BoneContainer.GetCompactPoseIndexFromSkeletonIndex(SkelIdx);
		if (CompactIdx == INDEX_NONE)
		{
			continue;
		}

		// Parent's component-space transform from the current evaluated pose.
		const FCompactPoseBoneIndex ParentCompact = Output.Pose.GetPose().GetParentBoneIndex(CompactIdx);
		FTransform ParentComp = FTransform::Identity;
		if (ParentCompact != INDEX_NONE)
		{
			ParentComp = Output.Pose.GetComponentSpaceTransform(ParentCompact);
		}

		const FTransform NewComp = SolvedLocal(SkelIdx) * ParentComp;
		OutBoneTransforms.Add(FBoneTransform(CompactIdx, NewComp));

		// Make this solved component transform visible to the next (child) chain
		// bone so the chain composes correctly within this single pass.
		Output.Pose.SetComponentSpaceTransform(CompactIdx, NewComp);
	}

	// Contract: ascending compact index.
	OutBoneTransforms.Sort(FCompareBoneTransformIndex());
}
