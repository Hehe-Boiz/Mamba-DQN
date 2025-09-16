let
    pkgs = import <nixpkgs> { 
        # config.rocmSupport = true; 
        config.allowUnfree = true;
        # config.rocm.gfxVer = [ "gfx1030" ];

        overlays = [ (import ./ale-py-overlay.nix) ];
    };

in 
pkgs.mkShell {
    packages = with pkgs; [
        (python312.withPackages (ps: with ps; [
            # torchWithRocm
            torch
            torchvision
            opencv-python

            autorom
            numpy
            pip
            requests
            matplotlib
            gymnasium
            gym
            ale-py
        ]))

        stdenv.cc.cc.lib

        rocmPackages.rocminfo
    ];

    # shellHook = ''
    #     # Set ROCm environment variables
    #     export ROCM_PATH=${pkgs.rocmPackages.rocm-runtime}
    #     export HIP_PATH=${pkgs.rocmPackages.hipcc}
    #     export HSA_PATH=${pkgs.rocmPackages.rccl}
    #     
    #     # Disable HIPBLASLT if needed (common ROCm issue)
    #     export TORCH_BLAS_PREFER_HIPBLASLT=0
    #     export HSA_OVERRIDE_GFX_VERSION=10.3.0
    #     export TORCH_USE_HIP_DSA=1
    #     
    #     echo "PyTorch ROCm environment ready!"
    # '';
}

