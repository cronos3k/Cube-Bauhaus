// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================

#include "CubeRigWorldAdapter.h"

#include "Components/CapsuleComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/OverlapResult.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Logging/LogMacros.h"

DEFINE_LOG_CATEGORY_STATIC(LogCubeRigWorld, Log, All);

namespace
{
	/**
	 * Convert a UCapsuleComponent into a world-space (A, B, radius) segment.
	 * UE capsules are defined by a half-height (along local +Z) and a radius,
	 * both scaled by the component's world scale. The segment endpoints are the
	 * centers of the two hemispheres: center +/- (half-height - radius) * upAxis.
	 */
	void CapsuleToSegmentWorld(const UCapsuleComponent& Cap, FVector& OutA, FVector& OutB, float& OutRadius)
	{
		const FTransform Xf = Cap.GetComponentTransform();
		const float Scale = Xf.GetMaximumAxisScale();
		const float Radius = Cap.GetUnscaledCapsuleRadius() * Scale;
		const float HalfHeight = Cap.GetUnscaledCapsuleHalfHeight() * Scale;
		const float SegHalf = FMath::Max(0.f, HalfHeight - Radius);

		const FVector Center = Xf.GetLocation();
		const FVector Up = Xf.GetUnitAxis(EAxis::Z); // capsule's local +Z in world.

		OutA = Center + Up * SegHalf;
		OutB = Center - Up * SegHalf;
		OutRadius = Radius;
	}

	/**
	 * Approximate an arbitrary primitive by its world bounds box: segment along
	 * the longest world axis through the center, radius = min of the other two
	 * half-extents. See header for the trade-off note.
	 */
	void BoundsToSegmentWorld(const FBoxSphereBounds& Bounds, FVector& OutA, FVector& OutB, float& OutRadius)
	{
		const FVector Center = Bounds.Origin;
		const FVector Ext = Bounds.BoxExtent; // half-extents along world X/Y/Z.

		// Pick the longest axis for the capsule's spine.
		int32 LongAxis = 0;
		if (Ext.Y > Ext.X && Ext.Y >= Ext.Z) LongAxis = 1;
		else if (Ext.Z > Ext.X && Ext.Z > Ext.Y) LongAxis = 2;

		FVector Dir = FVector::ZeroVector;
		Dir[LongAxis] = 1.f;
		const float SpineHalf = Ext[LongAxis];

		// Radius = smaller of the two remaining half-extents.
		const int32 A1 = (LongAxis + 1) % 3;
		const int32 A2 = (LongAxis + 2) % 3;
		const float Radius = FMath::Min(Ext[A1], Ext[A2]);

		OutA = Center + Dir * SpineHalf;
		OutB = Center - Dir * SpineHalf;
		OutRadius = FMath::Max(1.f, Radius); // never degenerate.
	}
}

void UCubeRigWorldAdapter::GatherObstacles(
	const UWorld* World,
	const USkeletalMeshComponent* SelfMesh,
	const FVector& Origin,
	float Radius,
	ECollisionChannel Channel,
	TArray<CrfCapsule>& Out)
{
	if (World == nullptr || SelfMesh == nullptr || Radius <= 0.f)
	{
		return;
	}

	const AActor* SelfOwner = SelfMesh->GetOwner();
	const FTransform MeshXf = SelfMesh->GetComponentTransform();

	// Overlap a sphere on the requested channel. Ignore the owning actor up front.
	FCollisionQueryParams Params(SCENE_QUERY_STAT(CubeRigGatherObstacles), /*bTraceComplex*/ false);
	if (SelfOwner)
	{
		Params.AddIgnoredActor(SelfOwner);
	}

	TArray<FOverlapResult> Overlaps;
	World->OverlapMultiByChannel(
		Overlaps,
		Origin,
		FQuat::Identity,
		Channel,
		FCollisionShape::MakeSphere(Radius),
		Params);

	for (const FOverlapResult& Hit : Overlaps)
	{
		UPrimitiveComponent* Prim = Hit.GetComponent();
		if (Prim == nullptr)
		{
			continue;
		}
		// Defensive: skip anything still belonging to ourselves (e.g. owner was
		// null so the ignore above did nothing).
		if (SelfOwner != nullptr && Prim->GetOwner() == SelfOwner)
		{
			continue;
		}

		FVector WorldA, WorldB;
		float WorldRadius = 0.f;

		if (const UCapsuleComponent* Cap = Cast<UCapsuleComponent>(Prim))
		{
			CapsuleToSegmentWorld(*Cap, WorldA, WorldB, WorldRadius);
		}
		else
		{
			BoundsToSegmentWorld(Prim->Bounds, WorldA, WorldB, WorldRadius);
		}

		// World (cm) -> SelfMesh component space, the solver's working space.
		const FVector LocalA = MeshXf.InverseTransformPosition(WorldA);
		const FVector LocalB = MeshXf.InverseTransformPosition(WorldB);
		// Radius is a length; only uniform component scale would change it. Apply
		// the inverse uniform scale so radii match the local-space coordinates.
		const float MeshScale = MeshXf.GetMaximumAxisScale();
		const float LocalRadius = (MeshScale > KINDA_SMALL_NUMBER) ? (WorldRadius / MeshScale) : WorldRadius;

		Out.Add(CubeRigIKMarshal::MakeCapsule(LocalA, LocalB, LocalRadius));
	}
}

void UCubeRigWorldAdapter::GatherSelfCollisionCapsules(
	const USkeletalMeshComponent* Mesh,
	const TArray<FName>& BoneNames,
	float Radius,
	TArray<CrfCapsule>& Out)
{
	if (Mesh == nullptr || BoneNames.Num() == 0)
	{
		return;
	}

	if (Mesh->GetSkeletalMeshAsset() == nullptr)
	{
		return;
	}
	const FReferenceSkeleton& RefSkel = Mesh->GetSkeletalMeshAsset()->GetRefSkeleton();

	for (const FName& BoneName : BoneNames)
	{
		const int32 BoneIdx = Mesh->GetBoneIndex(BoneName);
		if (BoneIdx == INDEX_NONE)
		{
			continue;
		}
		const int32 ParentIdx = RefSkel.GetParentIndex(BoneIdx);
		if (ParentIdx == INDEX_NONE)
		{
			continue; // root has no segment.
		}

		// Component-space transforms of this bone and its parent.
		const FVector ChildPos = Mesh->GetBoneTransform(BoneIdx, FTransform::Identity).GetLocation();
		const FVector ParentPos = Mesh->GetBoneTransform(ParentIdx, FTransform::Identity).GetLocation();

		Out.Add(CubeRigIKMarshal::MakeCapsule(ParentPos, ChildPos, Radius));
	}
}
