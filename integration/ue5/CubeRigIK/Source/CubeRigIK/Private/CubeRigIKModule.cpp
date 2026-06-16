// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
#include "CubeRigIKModule.h"

#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

#define LOCTEXT_NAMESPACE "FCubeRigIKModule"

void FCubeRigIKModule::StartupModule()
{
#if PLATFORM_WINDOWS
	// The DLL is delay-loaded (see Build.cs PublicDelayLoadDLLs); load it
	// explicitly from the plugin's staged ThirdParty/Win64 directory.
	const FString BaseDir = IPluginManager::Get().FindPlugin(TEXT("CubeRigIK"))->GetBaseDir();
	const FString DllPath = FPaths::Combine(
		BaseDir, TEXT("ThirdParty"), TEXT("CubeRigFFI"), TEXT("lib"), TEXT("Win64"),
		TEXT("cube_rig_ffi.dll"));
	FFILibraryHandle = FPlatformProcess::GetDllHandle(*DllPath);
	checkf(FFILibraryHandle != nullptr,
		TEXT("CubeRigIK: failed to load cube_rig_ffi.dll from %s"), *DllPath);
#endif
	// On Linux/Mac the .so/.dylib is linked directly (RPATH-resolved); no manual
	// load is required.
}

void FCubeRigIKModule::ShutdownModule()
{
	if (FFILibraryHandle != nullptr)
	{
		FPlatformProcess::FreeDllHandle(FFILibraryHandle);
		FFILibraryHandle = nullptr;
	}
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FCubeRigIKModule, CubeRigIK)
