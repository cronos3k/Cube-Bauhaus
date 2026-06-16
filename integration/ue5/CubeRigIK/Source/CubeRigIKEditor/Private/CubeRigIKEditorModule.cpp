// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// =============================================================================
//  UNCOMPILED-HERE SCAFFOLD (UE 5.4 plugin; cannot build in this repo).
// =============================================================================
//
// Minimal editor module entry point. The UAnimGraphNode_CubeRigIK UCLASS
// registers itself with the AnimGraph editor automatically (reflection), so this
// module only needs to exist to be loaded.

#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FDefaultModuleImpl, CubeRigIKEditor);
