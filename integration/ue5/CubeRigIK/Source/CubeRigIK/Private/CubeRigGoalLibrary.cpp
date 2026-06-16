// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================

#include "CubeRigGoalLibrary.h"

#include "Engine/HitResult.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"

bool UCubeRigGoalLibrary::TraceFootGoal(
	const UObject* WorldCtx,
	FVector FootWorld,
	float TraceUp,
	float TraceDown,
	FVector& OutGroundTarget)
{
	OutGroundTarget = FootWorld; // default: keep the animated foot height on a miss.

	const UWorld* World = GEngine ? GEngine->GetWorldFromContextObject(WorldCtx, EGetWorldErrorMode::ReturnNull) : nullptr;
	if (World == nullptr)
	{
		return false;
	}

	const FVector Up = FVector::UpVector;
	const FVector Start = FootWorld + Up * FMath::Max(0.f, TraceUp);
	const FVector End = FootWorld - Up * FMath::Max(0.f, TraceDown);

	FHitResult Hit;
	FCollisionQueryParams Params(SCENE_QUERY_STAT(CubeRigTraceFootGoal), /*bTraceComplex*/ true);
	if (World->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params))
	{
		OutGroundTarget = Hit.ImpactPoint;
		return true;
	}
	return false;
}

FVector UCubeRigGoalLibrary::ReachGoalFromActor(const AActor* TargetActor, FName Socket)
{
	if (TargetActor == nullptr)
	{
		return FVector::ZeroVector;
	}
	if (Socket.IsNone())
	{
		return TargetActor->GetActorLocation();
	}
	// GetActorLocation falls back gracefully if the socket is missing.
	return TargetActor->GetRootComponent()
		? TargetActor->GetRootComponent()->GetSocketLocation(Socket)
		: TargetActor->GetActorLocation();
}
