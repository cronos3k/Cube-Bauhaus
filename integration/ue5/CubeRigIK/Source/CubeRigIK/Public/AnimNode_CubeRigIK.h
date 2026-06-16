// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
//  See integration/ue5/README.md.
// =============================================================================
//
// FAnimNode_CubeRigIK — the production apply path. A skeletal-control AnimGraph
// node that, during animation evaluation, harvests the chain's bones from the
// component-space pose, calls crf_solve_chain_avoid, and writes the solved bone
// transforms back into the pose. Because it runs inside the anim eval it blends
// naturally with the rest of the AnimBP (alpha, layered blends, etc.).
//
// SPACE: this node works in the mesh's COMPONENT space, UE centimeters — the
// same single space the rest of the plugin uses. The effector target is taken
// in component space (convert from world before feeding the pin, or use the
// goal-library helpers + an inverse component transform on the AnimBP).

#pragma once

#include "CoreMinimal.h"
#include "BoneControllers/AnimNode_SkeletalControlBase.h"
#include "BoneContainer.h"
#include "AnimNode_CubeRigIK.generated.h"

struct CrfPrior;

/**
 * Inverse-kinematics control node backed by the cube-rig solver.
 *
 * THREAD SAFETY
 * -------------
 *  - EvaluateSkeletalControl_AnyThread runs on a worker thread. It must not
 *    touch UObjects or the game world. It only reads the pose + this node's
 *    cached/POD inputs and calls the (reentrant, allocation-local) C ABI.
 *  - The CrfPrior* is loaded lazily on the game thread path (InitializeBone
 *    references / first use) and is read-only afterwards, so sharing it on the
 *    worker thread is safe. It is freed when the node is destroyed.
 *  - Obstacle capsules cannot be gathered here (that needs the world, a
 *    game-thread query). Feed obstacles via the input array `Obstacles` which an
 *    upstream game-thread step (e.g. UCubeRigIKComponent or an AnimInstance
 *    NativeUpdate) fills in component space. Left empty => plain IK (matches
 *    crf_solve_chain).
 */
USTRUCT(BlueprintInternalUseOnly)
struct CUBERIGIK_API FAnimNode_CubeRigIK : public FAnimNode_SkeletalControlBase
{
	GENERATED_BODY()

	/** Root->tip bones to rotate. */
	UPROPERTY(EditAnywhere, Category = "Chain")
	TArray<FBoneReference> ChainBones;

	/** Bone whose head is driven to the target. */
	UPROPERTY(EditAnywhere, Category = "Chain")
	FBoneReference EffectorBone;

	/** Effector goal in COMPONENT space (pin: convert from world upstream). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Goal", meta = (PinShownByDefault))
	FVector EffectorTargetComponent = FVector::ZeroVector;

	/** Absolute path to a MotionPrior JSON. Empty => uniform compliance. */
	UPROPERTY(EditAnywhere, Category = "Solver")
	FString PriorJsonPath;

	/** DLS solve iterations (clamped >= 1). */
	UPROPERTY(EditAnywhere, Category = "Solver", meta = (PinHiddenByDefault, ClampMin = "1"))
	int32 Iterations = 16;

	/** Obstacle push-out iterations (clamped >= 1). */
	UPROPERTY(EditAnywhere, Category = "Solver", meta = (PinHiddenByDefault, ClampMin = "1"))
	int32 AvoidIterations = 8;

	/** Capsule radius for the chain's own bones during avoidance (cm). */
	UPROPERTY(EditAnywhere, Category = "Solver", meta = (PinHiddenByDefault))
	float BoneRadius = 4.f;

	/**
	 * Obstacle capsules in COMPONENT space, supplied by an upstream game-thread
	 * step. Plain C structs, not exposed as a pin; set from native code (e.g. an
	 * AnimInstance ThreadSafeUpdate that calls UCubeRigWorldAdapter). Empty =>
	 * plain IK.
	 */
	TArray<struct CrfCapsule> Obstacles;

	FAnimNode_CubeRigIK() = default;
	virtual ~FAnimNode_CubeRigIK();

	//~ Begin FAnimNode_SkeletalControlBase
	virtual void EvaluateSkeletalControl_AnyThread(
		FComponentSpacePoseContext& Output,
		TArray<FBoneTransform>& OutBoneTransforms) override;
	virtual bool IsValidToEvaluate(const USkeleton* Skeleton, const FBoneContainer& RequiredBones) override;
	virtual void InitializeBoneReferences(const FBoneContainer& RequiredBones) override;
	//~ End FAnimNode_SkeletalControlBase

private:
	/** Lazily load the prior (idempotent). Safe to call from the worker path
	 *  because PriorJsonPath is fixed and the handle is read-only after load. */
	void EnsurePriorLoaded();

	/** Cached prior handle (lazy). */
	CrfPrior* Prior = nullptr;

	/** True once we tried to load (so a failed load is not retried every frame). */
	bool bPriorLoadAttempted = false;
};
