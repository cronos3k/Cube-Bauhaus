// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
//  See integration/ue5/README.md.
// =============================================================================
//
// UCubeRigIKComponent — the easiest-to-run path. Drop it on a Character (or any
// actor with a SkeletalMeshComponent), name the chain bones + effector, point it
// at a target, and each frame it:
//   1. harvests the skeleton (parents, bind locals, chain, effector) from the
//      mesh's reference skeleton,
//   2. resolves the goal in component space,
//   3. gathers world + self obstacles as CrfCapsules,
//   4. calls crf_solve_chain_avoid,
//   5. applies the solved local rotations to the live pose.
//
// SPACE: everything is component space of TargetMesh, UE centimeters (see
// CubeRigIKMarshal.h). Obstacles and binds share that one space.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/EngineTypes.h" // ECollisionChannel
#include "CubeRigIKComponent.generated.h"

class USkeletalMeshComponent;
struct CrfPrior;

UENUM(BlueprintType)
enum class ECubeRigBindSource : uint8
{
	/** Use the skeletal mesh's reference (bind) pose. Stable; classic IK binds. */
	ReferencePose UMETA(DisplayName = "Reference Pose"),
	/** Use the current animated component-space pose each tick. Lets the solve
	 *  start from the live animation (additive-feel), at the cost of feedback if
	 *  you also write back to the same mesh — see the apply notes in the .cpp. */
	CurrentPose UMETA(DisplayName = "Current Pose"),
};

/**
 * Drop-on-a-Character IK driver backed by the cube-rig solver.
 *
 * The clean production apply path is the AnimGraph node (FAnimNode_CubeRigIK) or
 * a post-process AnimInstance. This component instead exposes the solved local
 * rotations via GetSolvedLocalRotation() AND can optionally write them straight
 * onto the mesh's component-space transforms each tick (bApplyToMeshDirectly).
 * The direct path is the simplest to see working; it runs after the anim update
 * and is overwritten next frame, so it composes poorly with a full AnimBP — use
 * the AnimNode for shipping. Trade-off documented at the apply site in the .cpp.
 */
UCLASS(ClassGroup = (Animation), meta = (BlueprintSpawnableComponent))
class CUBERIGIK_API UCubeRigIKComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UCubeRigIKComponent();

	//~ Begin UActorComponent
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
	//~ End UActorComponent

	// ── Mesh / skeleton ─────────────────────────────────────────────────────

	/** Mesh to drive. If null, auto-resolved from the owner on BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Skeleton")
	TObjectPtr<USkeletalMeshComponent> TargetMesh = nullptr;

	/** Root->tip bones to rotate, by name (e.g. upperarm_l, lowerarm_l). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Skeleton")
	TArray<FName> ChainBoneNames;

	/** Bone whose head is driven to the target (e.g. hand_l). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Skeleton")
	FName EffectorBoneName;

	/** Where the per-bone binds come from. See ECubeRigBindSource. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Skeleton")
	ECubeRigBindSource BindSource = ECubeRigBindSource::ReferencePose;

	// ── Solver params ───────────────────────────────────────────────────────

	/** Absolute path to a MotionPrior JSON (extract_prior). Empty => uniform. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Solver")
	FString PriorJsonPath;

	/** Capsule radius for the chain's own bones during avoidance (cm). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Solver")
	float BoneRadius = 4.f;

	/** DLS solve iterations (clamped >= 1). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Solver")
	int32 Iterations = 16;

	/** Obstacle push-out iterations (clamped >= 1). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Solver")
	int32 AvoidIterations = 8;

	// ── Goal source ─────────────────────────────────────────────────────────

	/** Explicit world-space target. Used when TargetActor is null. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Goal")
	FVector EffectorTargetWorld = FVector::ZeroVector;

	/** If set, the goal is this actor's TargetSocket (overrides EffectorTargetWorld). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Goal")
	TObjectPtr<AActor> TargetActor = nullptr;

	/** Socket/bone on TargetActor's root component; NAME_None => actor origin. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Goal")
	FName TargetSocket = NAME_None;

	// ── Obstacles ───────────────────────────────────────────────────────────

	/** Gather nearby world primitives as capsule obstacles each tick. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Obstacles")
	bool bGatherWorldObstacles = true;

	/** Sphere radius (cm) around the effector for the obstacle overlap query. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Obstacles")
	float ObstacleQueryRadius = 150.f;

	/** Collision channel for the obstacle overlap query. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Obstacles")
	TEnumAsByte<ECollisionChannel> ObstacleChannel = ECC_WorldStatic;

	/** Bones treated as self-collision capsules (e.g. spine/torso). Optional. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Obstacles")
	TArray<FName> SelfCollisionBones;

	// ── Apply ───────────────────────────────────────────────────────────────

	/**
	 * Write the solved rotations straight onto the mesh this tick. Simplest path
	 * to see it working; not recommended for shipping (see class comment). When
	 * false, the result is only published via GetSolvedLocalRotation() for a
	 * post-process AnimInstance / the AnimNode to consume.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cube Rig IK|Apply")
	bool bApplyToMeshDirectly = true;

	// ── Read-back API (for a post-process AnimInstance) ─────────────────────

	/** Solved local rotation for a mesh bone index, or identity if not solved. */
	UFUNCTION(BlueprintCallable, Category = "Cube Rig IK")
	FQuat GetSolvedLocalRotation(int32 MeshBoneIndex) const;

	/** True if the last tick produced a valid solve. */
	UFUNCTION(BlueprintCallable, Category = "Cube Rig IK")
	bool HasValidSolve() const { return bHasValidSolve; }

private:
	/** Resolve TargetMesh + bone indices + prior. Returns true if ready to tick. */
	bool ResolveSetup();

	/** Free the cached prior handle. */
	void ReleasePrior();

	// ── Cached, resolved-once state ─────────────────────────────────────────

	/** Cached prior handle (loaded once on BeginPlay), or null for uniform. */
	CrfPrior* Prior = nullptr;

	/** Mesh bone indices for the chain, in order. INDEX_NONE if unresolved. */
	TArray<int32> ChainBoneIndices;

	/** Mesh bone index of the effector. INDEX_NONE if unresolved. */
	int32 EffectorBoneIndex = INDEX_NONE;

	/** True once ResolveSetup succeeded; gates TickComponent. */
	bool bSetupValid = false;

	/** True if the most recent solve succeeded (read-back guard). */
	bool bHasValidSolve = false;

	/**
	 * Last solved local rotation per mesh bone (length == mesh bone count).
	 * Published for read-back; also consumed by the direct-apply path.
	 */
	TArray<FQuat> SolvedLocalRotations;
};
