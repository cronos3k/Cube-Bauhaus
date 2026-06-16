// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
#pragma once

#include "Modules/ModuleManager.h"

/**
 * Runtime module for the CubeRigIK plugin. On Win64 it explicitly loads the
 * delay-loaded cube_rig_ffi DLL from the plugin's staged ThirdParty path so the
 * Blueprint library can call into it.
 */
class FCubeRigIKModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	/** Handle to the dynamically loaded cube_rig_ffi shared library (Win64). */
	void* FFILibraryHandle = nullptr;
};
