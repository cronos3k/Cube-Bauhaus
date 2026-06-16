// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
//  See integration/ue5/README.md.
// =============================================================================
//
// Goal hooks: small Blueprint-callable helpers that turn LIVE gameplay/world
// data into an effector target (an FVector) for the IK solver. These are the
// "what does the hand/foot want to reach this frame" producers — the component
// or AnimNode then converts the chosen WORLD goal into the solver's component
// space and solves toward it.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CubeRigGoalLibrary.generated.h"

class AActor;

/**
 * Blueprint-callable producers of effector goals (all outputs in WORLD space;
 * convert into the mesh's component space before solving).
 */
UCLASS()
class CUBERIGIK_API UCubeRigGoalLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Foot-placement goal: line-trace from above `FootWorld` down to the ground
	 * and report the hit point. Feeds gameplay (the animated foot position) into
	 * the solver so the foot plants on uneven terrain instead of clipping/floating.
	 *
	 * Traces from (FootWorld + Up*TraceUp) to (FootWorld - Up*TraceDown) on the
	 * Visibility channel. On a hit, OutGroundTarget is the impact point; on a
	 * miss it is left at FootWorld (so the foot keeps its animated height).
	 *
	 * @param WorldCtx        any UObject with a world (e.g. the mesh component).
	 * @param FootWorld       current animated foot location (world cm).
	 * @param TraceUp         how far above the foot to start (cm).
	 * @param TraceDown       how far below the foot to trace (cm).
	 * @param OutGroundTarget receives the world-space goal.
	 * @return true if the ground was hit.
	 */
	UFUNCTION(BlueprintCallable, Category = "Cube Rig IK|Goals", meta = (WorldContext = "WorldCtx"))
	static bool TraceFootGoal(
		const UObject* WorldCtx,
		FVector FootWorld,
		float TraceUp,
		float TraceDown,
		FVector& OutGroundTarget);

	/**
	 * Hand-reach goal: world-space location of a socket (or actor origin) on a
	 * target actor. Feeds a gameplay object's position (a lever, a held prop, an
	 * NPC's hand) into the solver as the reach target.
	 *
	 * @param TargetActor the actor to reach toward (null => ZeroVector).
	 * @param Socket       socket/bone on the target's root component; NAME_None
	 *                     => the actor's location.
	 * @return world-space reach goal.
	 */
	UFUNCTION(BlueprintCallable, Category = "Cube Rig IK|Goals")
	static FVector ReachGoalFromActor(const AActor* TargetActor, FName Socket);
};
