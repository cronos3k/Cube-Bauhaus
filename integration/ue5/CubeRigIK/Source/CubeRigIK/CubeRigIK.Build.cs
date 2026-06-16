// Copyright Cube Bauhaus. SPDX-License-Identifier: MIT
//
// Module rules for the CubeRigIK runtime module. This links the prebuilt
// `cube_rig_ffi` library (built from crates/cube-rig-ffi) and exposes its C
// header to the C++ module.
//
// Drop the built library here before compiling the plugin (see
// integration/ue5/README.md for the full runbook):
//
//   ThirdParty/CubeRigFFI/include/cube_rig.h                    (the C header)
//   ThirdParty/CubeRigFFI/lib/Win64/cube_rig_ffi.dll.lib        (import lib)
//   ThirdParty/CubeRigFFI/lib/Win64/cube_rig_ffi.dll            (runtime DLL)
//   ThirdParty/CubeRigFFI/lib/Linux/libcube_rig_ffi.so          (shared object)
//   ThirdParty/CubeRigFFI/lib/Mac/libcube_rig_ffi.dylib         (dylib)
//
// You may instead link the static lib (libcube_rig_ffi.a / cube_rig_ffi.lib);
// see the commented branch below.

using System.IO;
using UnrealBuildTool;

public class CubeRigIK : ModuleRules
{
	public CubeRigIK(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
		});

		// ── Third-party cube-rig-ffi ────────────────────────────────────────
		string ThirdParty = Path.Combine(ModuleDirectory, "..", "..", "ThirdParty", "CubeRigFFI");
		string IncludeDir = Path.Combine(ThirdParty, "include");
		string LibRoot    = Path.Combine(ThirdParty, "lib");

		PublicIncludePaths.Add(IncludeDir);

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			// Link against the DLL's import library, stage the DLL at runtime.
			string Win64 = Path.Combine(LibRoot, "Win64");
			PublicAdditionalLibraries.Add(Path.Combine(Win64, "cube_rig_ffi.dll.lib"));

			string Dll = "cube_rig_ffi.dll";
			RuntimeDependencies.Add(Path.Combine(Win64, Dll));
			PublicDelayLoadDLLs.Add(Dll);

			// Static-lib alternative (no DLL to stage):
			// PublicAdditionalLibraries.Add(Path.Combine(Win64, "cube_rig_ffi.lib"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			string Linux = Path.Combine(LibRoot, "Linux");
			string So = Path.Combine(Linux, "libcube_rig_ffi.so");
			PublicAdditionalLibraries.Add(So);
			RuntimeDependencies.Add(So);

			// Static-lib alternative:
			// PublicAdditionalLibraries.Add(Path.Combine(Linux, "libcube_rig_ffi.a"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			string Mac = Path.Combine(LibRoot, "Mac");
			string Dylib = Path.Combine(Mac, "libcube_rig_ffi.dylib");
			PublicAdditionalLibraries.Add(Dylib);
			RuntimeDependencies.Add(Dylib);

			// Static-lib alternative:
			// PublicAdditionalLibraries.Add(Path.Combine(Mac, "libcube_rig_ffi.a"));
		}
	}
}
