// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
//  See integration/ue5/README.md.
// =============================================================================
//
// World -> capsule adapter. This is the part that turns LIVE world/gameplay
// state into solver input: it queries the physics world for nearby primitives
// and converts them into the CrfCapsule obstacles that crf_solve_chain_avoid
// pushes the chain out of. It also builds self-collision capsules from the
// character's own bones (e.g. so a reaching arm avoids the torso).
//
// SPACE CONTRACT: every capsule produced here is expressed in the
// SkeletalMeshComponent's COMPONENT space (UE centimeters), the SAME space the
// binds + target are packed in (see CubeRigIKMarshal.h). World hits are pulled
// back through the mesh's component transform before being emitted.

#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h" // ECollisionChannel
#include "Kismet/BlueprintFunctionLibrary.h"

// CrfCapsule lives in the C ABI; pull it (and the marshal helpers) in.
#include "CubeRigIKMarshal.h"

#include "CubeRigWorldAdapter.generated.h"

class UWorld;
class USkeletalMeshComponent;

/**
 * Static helpers that gather obstacle capsules for the IK solver.
 *
 * Not Blueprint-exposed for the CrfCapsule overloads (CrfCapsule is a plain C
 * struct, not a USTRUCT). These are intended to be called from C++ (the
 * UCubeRigIKComponent / AnimNode). A Blueprint-friendly debug-draw wrapper could
 * be layered on top if needed.
 */
UCLASS()
class CUBERIGIK_API UCubeRigWorldAdapter : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Overlap a sphere of `Radius` (cm) around `Origin` (WORLD space) on
	 * `Channel`, and append a CrfCapsule for every hit primitive that does NOT
	 * belong to `SelfMesh`'s owner. Capsules are emitted in `SelfMesh`'s
	 * COMPONENT space.
	 *
	 * Conversion rules per primitive:
	 *  - UCapsuleComponent: use its actual segment (the two hemisphere centers)
	 *    and scaled radius directly — exact.
	 *  - Anything else: approximate from the component's world-space bounding box
	 *    (FBoxSphereBounds). The capsule segment runs along the box's longest
	 *    world axis through the box center, spanning +/- half that extent; the
	 *    capsule radius is the smaller of the remaining two half-extents. This
	 *    over-covers thin/long props reasonably and under-covers very boxy props
	 *    (acceptable: obstacle avoidance is a soft push-out, not exact collision).
	 *
	 * @param World     world to query (usually SelfMesh->GetWorld()).
	 * @param SelfMesh  the character's mesh; its owner is excluded and its
	 *                  component transform defines the output space.
	 * @param Origin    query center in WORLD space (e.g. effector world position).
	 * @param Radius    query + bounds radius in cm.
	 * @param Channel   collision channel to overlap.
	 * @param Out       capsules are APPENDED (caller may pre-reserve / reuse).
	 */
	static void GatherObstacles(
		const UWorld* World,
		const USkeletalMeshComponent* SelfMesh,
		const FVector& Origin,
		float Radius,
		ECollisionChannel Channel,
		TArray<CrfCapsule>& Out);

	/**
	 * Build one capsule per named bone segment from the mesh's CURRENT bone
	 * transforms (component space), for self-collision avoidance (arm vs torso,
	 * etc.). Each capsule spans from a bone to its parent in `BoneNames`; bones
	 * whose parent is not resolvable are skipped.
	 *
	 * Pass the bones you want to AVOID here (e.g. spine/torso bones), NOT the
	 * chain bones being solved — feeding a chain its own current capsules would
	 * fight the solve.
	 *
	 * @param Mesh       mesh to read current pose from (component space output).
	 * @param BoneNames  bones to turn into segment capsules.
	 * @param Radius     capsule radius in cm.
	 * @param Out        capsules are APPENDED.
	 */
	static void GatherSelfCollisionCapsules(
		const USkeletalMeshComponent* Mesh,
		const TArray<FName>& BoneNames,
		float Radius,
		TArray<CrfCapsule>& Out);
};
