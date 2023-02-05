(* ::Package:: *)

(*
See also https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor 
for more details on compiler and linker options.
*)
Switch[$OperatingSystem,
  "MacOSX", (* Compilation settings for OS X *)
  {
	"CompileOptions" -> {
		" -Wall"
		,"-Wextra"
		,"-Wno-unused-parameter"
		,"-mmacosx-version-min=12.0"
		,"-std=c++17"
		,"-Xpreprocessor -fopenmp -Xpreprocessor -fopenmp-simd"
		,"-fno-math-errno"
		,Switch[$SystemID
			,"MacOSX-ARM64"
			(*,"-mcpu=apple-m1 -mtune=native -framework Accelerate"*)
			,"-mcpu=apple-m1 -mtune=native -framework Accelerate"
		
			,"MacOSX-x86-64"
			,"-march=native -mtune=native -fveclib=SVML"
		
			,_,""
		]
		,"-ffast-math"
		,"-Ofast"
		,"-flto"
		,"-gline-tables-only"
		,"-gcolumn-info"
(*		,"-foptimization-record-file="<>FileNameJoin[{$HomeDirectory,"RepulsionLink_OptimizationRecord.txt"}]
		,"-Rpass-analysis=loop-distribute"
		,"-Rpass-analysis=loop-vectorize"
		,"-Rpass-missed=loop-vectorize"
		,"-Rpass=loop-vectorize"*)
		}
	,"LinkerOptions"->{
		"-lm"
		,"-ldl"
		,"-lomp"
		}
    ,"IncludeDirectories" -> {
		$OpenMPIncludeDirectory
		,"/opt/local/include"
		,$RepulsorIncludeDirectory
	    }
    ,"LibraryDirectories" -> {
        $OpenMPLibraryDirectory
        ,"/opt/local/lib"
	    (*,FileNameJoin[{$InstallationDirectory,"SystemFiles","Libraries",$SystemID}]*)
	    }
    (*,"ShellCommandFunction" -> Print*)
	,"ShellOutputFunction" -> Print
 },

  "Unix", (* Compilation settings for Linux *)
  {
	"CompileOptions" -> {" -O3"," -Wall -std=c++17 -DMKL_ILP64 -m64 -fopenmp -march=native"}
	,"LinkerOptions"->{" -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -ltbb"}
    ,"IncludeDirectories" -> { "/opt/intel/oneapi/mkl/latest/include","/opt/intel/oneapi/compiler/latest/mac/compiler/include", "/opt/local/include"}
    ,"LibraryDirectories" -> {"/opt/intel/oneapi/mkl/latest/lib","/opt/intel/oneapi/compiler/latest/mac/compiler/lib",FileNameJoin[{$InstallationDirectory,"SystemFiles","Libraries",$SystemID}]}
    ,"ShellOutputFunction" -> Print
  },

  "Windows", (* Compilation settings for Windows *)
  {
    "CompileOptions" -> {"/EHsc", "/wd4244", "/DNOMINMAX", "/DMKL_ILP64","/arch:AVX"}
	,"LinkerOptions"->{" mkl_intel_ilp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib"}
    ,"IncludeDirectories" -> { $OpenMPIncludeDirectory}
    ,"LibraryDirectories" -> {FileNameJoin[{$InstallationDirectory,"SystemFiles","Libraries",$SystemID}]}
    ,"ShellOutputFunction" -> Print
  }
]
