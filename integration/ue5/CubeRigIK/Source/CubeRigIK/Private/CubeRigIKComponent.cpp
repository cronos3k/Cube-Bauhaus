// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================

#include "CubeRigIKComponent.h"

#include "CubeRigIKMarshal.h"
#include "CubeRigWorldAdapter.h"

#include "Components/SkeletalMeshComponent.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Logging/LogMacros.h"

DEFINE_LOG_CATEGORY_STATIC(LogCubeRigIKComp, Log, All);

UCubeRigIKComponent::UCubeRigIKComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	// Tick AFTER the mesh's pose so current-pose binds / direct apply see the
	// up-to-date animated transforms. (TG_PostPhysics keeps world overlaps using
	// settled physics positions, too.)
	PrimaryComponentTick.TickGroup = TG_PostPhysics;
}

void UCubeRigIKComponent::BeginPlay()
{
	Super::BeginPlay();
	bSetupValid = ResolveSetup();
}

void UCubeRigIKComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	ReleasePrior();
	Super::EndPlay(EndPlayReason);
}

void UCubeRigIKComponent::ReleasePrior()
{
	if (Prior != nullptr)
	{
		crf_prior_free(Prior);
		Prior = nullptr;
	}
}

bool UCubeRigIKComponent::ResolveSetup()
{
	// ── Resolve the mesh ────────────────────────────────────────────────────
	if (TargetMesh == nullptr)
	{
		if (AActor* Owner = GetOwner())
		{
			TargetMesh = Owner->FindComponentByClass<USkeletalMeshComponent>();
		}
	}
	if (TargetMesh == nullptr || TargetMesh->GetSkeletalMeshAsset() == nullptr)
	{
		UE_LOG(LogCubeRigIKComp, Warning, TEXT("CubeRigIK: no skeletal mesh on %s; disabled."),
			*GetNameSafe(GetOwner()));
		return false;
	}

	// ── Resolve chain + effector bone indices against the mesh ──────────────
	ChainBoneIndices.Reset();
	for (const FName& Name : ChainBoneNames)
	{
		const int32 Idx = TargetMesh->GetBoneIndex(Name);
		if (Idx == INDEX_NONE)
		{
			UE_LOG(LogCubeRigIKComp, Warning, TEXT("CubeRigIK: chain bone '%s' not found; disabled."), *Name.ToString());
			return false;
		}
		ChainBoneIndices.Add(Idx);
	}
	if (ChainBoneIndices.Num() == 0)
	{
		UE_LOG(LogCubeRigIKComp, Warning, TEXT("CubeRigIK: empty chain; disabled."));
		return false;
	}

	EffectorBoneIndex = TargetMesh->GetBoneIndex(EffectorBoneName);
	if (EffectorBoneIndex == INDEX_NONE)
	{
		UE_LOG(LogCubeRigIKComp, Warning, TEXT("CubeRigIK: effector bone '%s' not found; disabled."), *EffectorBoneName.ToString());
		return false;
	}

	// ── Load the prior once ─────────────────────────────────────────────────
	ReleasePrior();
	if (!PriorJsonPath.IsEmpty())
	{
		Prior = crf_prior_load_json(TCHAR_TO_UTF8(*PriorJsonPath));
		if (Prior == nullptr)
		{
			UE_LOG(LogCubeRigIKComp, Warning, TEXT("CubeRigIK: failed to load prior '%s'; using uniform."), *PriorJsonPath);
		}
	}

	const int32 BoneCount = TargetMesh->GetNumBones();
	SolvedLocalRotations.Init(FQuat::Identity, BoneCount);
	bHasValidSolve = false;
	return true;
}

void UCubeRigIKComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// Null/again-safety: bail gracefully if anything is missing.
	if (!bSetupValid || TargetMesh == nullptr || TargetMesh->GetSkeletalMeshAsset() == nullptr)
	{
		bHasValidSolve = false;
		return;
	}

	const USkeletalMesh* MeshAsset = TargetMesh->GetSkeletalMeshAsset();
	const FReferenceSkeleton& RefSkel = MeshAsset->GetRefSkeleton();
	const int32 BoneCount = RefSkel.GetNum();
	if (BoneCount == 0)
	{
		bHasValidSolve = false;
		return;
	}

	// =========================================================================
	// 1. HARVEST SKELETON STATE  (parents[], local_bind[], chain[], effector)
	// =========================================================================
	//
	// We feed the solver LOCAL (parent-relative) bind transforms — that is what
	// the C ABI's `local_bind` means and what its internal forward-kinematics
	// expects. Two sources:
	//   ReferencePose: RefSkeleton's per-bone local rest transforms. Stable,
	//     authoring-time; the classic IK bind. Recommended.
	//   CurrentPose: the mesh's current LOCAL bone-space transforms this frame
	//     (GetBoneSpaceTransforms), so the solve starts from the live animation.
	//
	// Parent indices come straight from the RefSkeleton and are already
	// topologically ordered (UE guarantees parent index < child index), which is
	// exactly what crf_solve_chain requires.

	TArray<uint64> CParents;
	CParents.SetNumUninitialized(BoneCount);

	TArray<float> CLocalBind;
	CLocalBind.SetNumUninitialized(BoneCount * 16);

	// Source of local transforms.
	TArray<FTransform> LocalPose;
	if (BindSource == ECubeRigBindSource::CurrentPose)
	{
		// Current LOCAL (bone-space) transforms of the live pose.
		LocalPose = TargetMesh->GetBoneSpaceTransforms();
		if (LocalPose.Num() != BoneCount)
		{
			// Pose not ready yet this frame (e.g. before first anim eval).
			bHasValidSolve = false;
			return;
		}
	}

	for (int32 i = 0; i < BoneCount; ++i)
	{
		const int32 Parent = RefSkel.GetParentIndex(i);
		CParents[i] = (Parent == INDEX_NONE) ? CRF_NO_PARENT : static_cast<uint64>(Parent);

		const FTransform& Local = (BindSource == ECubeRigBindSource::CurrentPose)
			? LocalPose[i]
			: RefSkel.GetRefBonePose()[i]; // reference-pose local transform.

		CubeRigIKMarshal::PackTransform(Local, &CLocalBind[i * 16]);
	}

	TArray<uint64> CChain;
	CChain.SetNumUninitialized(ChainBoneIndices.Num());
	for (int32 i = 0; i < ChainBoneIndices.Num(); ++i)
	{
		CChain[i] = static_cast<uint64>(ChainBoneIndices[i]);
	}

	// =========================================================================
	// 2. RESOLVE THE GOAL  (world -> component space of TargetMesh)
	// =========================================================================
	FVector GoalWorld = EffectorTargetWorld;
	if (TargetActor != nullptr)
	{
		GoalWorld = TargetSocket.IsNone()
			? TargetActor->GetActorLocation()
			: (TargetActor->GetRootComponent()
				? TargetActor->GetRootComponent()->GetSocketLocation(TargetSocket)
				: TargetActor->GetActorLocation());
	}

	const FTransform MeshXf = TargetMesh->GetComponentTransform();
	const FVector GoalLocal = MeshXf.InverseTransformPosition(GoalWorld);
	float CTarget[3];
	CubeRigIKMarshal::PackVector(GoalLocal, CTarget);

	// =========================================================================
	// 3. GATHER OBSTACLES  (world + self-collision -> CrfCapsule[], component space)
	// =========================================================================
	TArray<CrfCapsule> Obstacles;
	if (bGatherWorldObstacles)
	{
		// Query around the effector's CURRENT world position so we capture what's
		// near the hand/foot, not the character root.
		const FVector EffectorWorld = TargetMesh->GetBoneTransform(EffectorBoneIndex).GetLocation();
		UCubeRigWorldAdapter::GatherObstacles(
			GetWorld(), TargetMesh, EffectorWorld, ObstacleQueryRadius, ObstacleChannel, Obstacles);
	}
	if (SelfCollisionBones.Num() > 0)
	{
		UCubeRigWorldAdapter::GatherSelfCollisionCapsules(
			TargetMesh, SelfCollisionBones, BoneRadius, Obstacles);
	}

	// =========================================================================
	// 4. SOLVE  (crf_solve_chain_avoid)
	// =========================================================================
	TArray<float> COut;
	COut.SetNumZeroed(BoneCount * 4);

	const int Code = crf_solve_chain_avoid(
		CParents.GetData(),
		CLocalBind.GetData(),
		static_cast<size_t>(BoneCount),
		CChain.GetData(),
		static_cast<size_t>(CChain.Num()),
		static_cast<uint64>(EffectorBoneIndex),
		CTarget,
		Prior,
		nullptr,                                   // v1: no hinge limits.
		FMath::Max(1, Iterations),
		Obstacles.Num() > 0 ? Obstacles.GetData() : nullptr,
		static_cast<size_t>(Obstacles.Num()),
		BoneRadius,
		FMath::Max(1, AvoidIterations),
		COut.GetData());

	if (Code != CRF_OK)
	{
		UE_LOG(LogCubeRigIKComp, Verbose, TEXT("CubeRigIK: solver returned %d"), Code);
		bHasValidSolve = false;
		return;
	}

	// Publish solved local rotations for read-back (post-process AnimInstance).
	if (SolvedLocalRotations.Num() != BoneCount)
	{
		SolvedLocalRotations.Init(FQuat::Identity, BoneCount);
	}
	for (int32 i = 0; i < BoneCount; ++i)
	{
		SolvedLocalRotations[i] = CubeRigIKMarshal::UnpackQuat(COut.GetData(), i);
	}
	bHasValidSolve = true;

	// =========================================================================
	// 5. APPLY
	// =========================================================================
	//
	// APPLY-PATH TRADE-OFF
	// --------------------
	// The solver returns per-bone LOCAL rotations. There are three places to put
	// them back:
	//   (a) AnimGraph node (FAnimNode_CubeRigIK)  -- the PROPER path: runs inside
	//       the anim eval, composes with the rest of the AnimBP, thread-safe.
	//   (b) post-process AnimInstance reading GetSolvedLocalRotation() -- also
	//       clean; this component just produces data.
	//   (c) direct write onto the mesh here (below) -- simplest to SEE working,
	//       but it runs AFTER the anim update and is clobbered next frame, and it
	//       does not blend with the AnimBP. Use only for a smoke test or for a
	//       mesh with no AnimBP driving the chain.
	//
	// We implement (c) behind bApplyToMeshDirectly and leave (a)/(b) as the
	// recommended routes. The direct path sets only the CHAIN bones' local
	// rotations via the component-space transform buffer, preserving translation
	// and scale from the current local pose.
	if (bApplyToMeshDirectly)
	{
		// Work in LOCAL space: combine the solved rotation with the existing
		// local translation/scale, then let the mesh recompute component space.
		const TArray<FTransform>& CurrentLocal =
			(LocalPose.Num() == BoneCount) ? LocalPose : TargetMesh->GetBoneSpaceTransforms();

		auto ApplyBone = [&](int32 BoneIdx)
		{
			if (!CurrentLocal.IsValidIndex(BoneIdx))
			{
				return;
			}
			FTransform NewLocal = CurrentLocal[BoneIdx];
			NewLocal.SetRotation(SolvedLocalRotations[BoneIdx]);
			// SetBoneTransformByName works in component space; convert local->comp
			// using the parent's component transform.
			const int32 Parent = RefSkel.GetParentIndex(BoneIdx);
			const FTransform ParentComp = (Parent != INDEX_NONE)
				? TargetMesh->GetBoneTransform(Parent, FTransform::Identity)
				: FTransform::Identity;
			const FTransform CompXf = NewLocal * ParentComp;
			const FName BoneName = RefSkel.GetBoneName(BoneIdx);
			TargetMesh->SetBoneTransformByName(BoneName, CompXf, EBoneSpaces::ComponentSpace);
		};

		// Apply parents before children so each child sees its updated parent.
		for (int32 BoneIdx : ChainBoneIndices)
		{
			ApplyBone(BoneIdx);
		}
	}
}

FQuat UCubeRigIKComponent::GetSolvedLocalRotation(int32 MeshBoneIndex) const
{
	return SolvedLocalRotations.IsValidIndex(MeshBoneIndex)
		? SolvedLocalRotations[MeshBoneIndex]
		: FQuat::Identity;
}
