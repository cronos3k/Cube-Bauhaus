// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================
//
// Editor-only module: registers the AnimGraph editor node
// (UAnimGraphNode_CubeRigIK) that surfaces FAnimNode_CubeRigIK in an Animation
// Blueprint's AnimGraph. Loaded UncookedOnly so it ships in editor/uncooked
// builds but is stripped from cooked game builds.

using UnrealBuildTool;

public class CubeRigIKEditor : ModuleRules
{
	public CubeRigIKEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"AnimGraph",       // UAnimGraphNode_SkeletalControlBase
			"BlueprintGraph",  // K2 node plumbing AnimGraph builds on
			"UnrealEd",        // editor module base
			"CubeRigIK",       // the runtime module with FAnimNode_CubeRigIK
		});
	}
}
