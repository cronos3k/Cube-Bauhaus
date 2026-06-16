// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================

#include "AnimGraphNode_CubeRigIK.h"

#define LOCTEXT_NAMESPACE "CubeRigIKEditor"

FText UAnimGraphNode_CubeRigIK::GetControllerDescription() const
{
	return LOCTEXT("CubeRigIKController", "Cube Rig IK");
}

FText UAnimGraphNode_CubeRigIK::GetTooltipText() const
{
	return LOCTEXT("CubeRigIKTooltip",
		"Inverse kinematics powered by the cube-rig solver (cube-rig-ffi). Drives "
		"the effector bone to a component-space target, with optional motion prior "
		"and capsule obstacle avoidance.");
}

FText UAnimGraphNode_CubeRigIK::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
	return GetControllerDescription();
}

FString UAnimGraphNode_CubeRigIK::GetNodeCategory() const
{
	return TEXT("Cube Rig IK");
}

#undef LOCTEXT_NAMESPACE
