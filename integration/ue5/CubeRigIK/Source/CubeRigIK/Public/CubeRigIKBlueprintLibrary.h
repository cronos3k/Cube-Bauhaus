// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CubeRigIKBlueprintLibrary.generated.h"

/**
 * Blueprint-callable wrapper over the cube-rig IK solver (cube-rig-ffi C ABI).
 *
 * Marshals UE container/value types into the flat C arrays `crf_solve_chain`
 * expects and returns the solved per-bone local rotations.
 *
 * Coordinate-space note: cube-rig is engine-agnostic and works in whatever space
 * you feed it. Pass bone binds, parents and the target in a CONSISTENT space
 * (e.g. component space of your SkeletalMeshComponent). The solver does not
 * convert handedness or units for you.
 */
UCLASS()
class CUBERIGIK_API UCubeRigIKBlueprintLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Solve an IK chain so the effector bone reaches Target.
	 *
	 * @param BoneBinds     Per-bone bind-pose transforms (component space), one per bone.
	 * @param Parents       Per-bone parent index, or -1 for a root. Must be topologically
	 *                      ordered (parent index < child index). Length == BoneBinds.
	 * @param Chain         Root->tip bone indices to rotate.
	 * @param Effector      Bone whose head is driven to the target (often a descendant
	 *                      of the last chain bone).
	 * @param Target        Goal position in the same space as BoneBinds.
	 * @param PriorJsonPath Optional absolute path to a MotionPrior JSON (from
	 *                      `extract_prior`); empty => uniform compliance.
	 * @param Iterations    DLS iterations (clamped to >= 1).
	 * @param OutRotations  Receives the solved local rotation of EVERY bone (length ==
	 *                      BoneBinds); non-chain bones keep their bind rotation.
	 * @return true on success; false on a bad argument / solver error (OutRotations
	 *         is left empty on failure).
	 */
	UFUNCTION(BlueprintCallable, Category = "Cube Rig IK",
		meta = (AutoCreateRefTerm = "PriorJsonPath"))
	static bool SolveArmIK(
		const TArray<FTransform>& BoneBinds,
		const TArray<int32>& Parents,
		const TArray<int32>& Chain,
		int32 Effector,
		FVector Target,
		const FString& PriorJsonPath,
		int32 Iterations,
		TArray<FQuat>& OutRotations);

	/** Returns the linked cube-rig-ffi library version string. */
	UFUNCTION(BlueprintCallable, Category = "Cube Rig IK")
	static FString GetCubeRigVersion();
};
