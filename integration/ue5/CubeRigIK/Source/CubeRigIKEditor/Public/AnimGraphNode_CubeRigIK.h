// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
//  See integration/ue5/README.md.
// =============================================================================
//
// Editor wrapper that exposes FAnimNode_CubeRigIK as a node in an Animation
// Blueprint's AnimGraph. UAnimGraphNode_SkeletalControlBase provides the pin
// generation, preview, and title plumbing; we only supply the runtime node and
// some display strings.

#pragma once

#include "CoreMinimal.h"
#include "AnimGraphNode_SkeletalControlBase.h"
#include "AnimNode_CubeRigIK.h"
#include "AnimGraphNode_CubeRigIK.generated.h"

UCLASS(MinimalAPI)
class UAnimGraphNode_CubeRigIK : public UAnimGraphNode_SkeletalControlBase
{
	GENERATED_BODY()

public:
	/** The runtime node this editor node edits. */
	UPROPERTY(EditAnywhere, Category = "Settings")
	FAnimNode_CubeRigIK Node;

	//~ Begin UEdGraphNode
	virtual FText GetNodeTitle(ENodeTitleType::Type TitleType) const override;
	virtual FText GetTooltipText() const override;
	//~ End UEdGraphNode

	//~ Begin UAnimGraphNode_Base
	virtual FString GetNodeCategory() const override;
	//~ End UAnimGraphNode_Base

protected:
	//~ Begin UAnimGraphNode_SkeletalControlBase
	virtual FText GetControllerDescription() const override;
	virtual const FAnimNode_SkeletalControlBase* GetNode() const override { return &Node; }
	//~ End UAnimGraphNode_SkeletalControlBase
};
