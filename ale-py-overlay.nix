final: prev: {
    atari-roms = prev.stdenv.mkDerivation rec {
        pname = "atari-roms";
        version = "0.6.1";
        src = prev.lib.cleanSource ./.;
          
        nativeBuildInputs = [ prev.python312Packages.autorom ];

        installPhase = ''
            mkdir $out
            AutoROM --accept-license --install-dir $out
        '';

        outputHashAlgo = "sha256";
        outputHashMode = "recursive";
        outputHash = "sha256-XPJtzcbCrs0uw0pCKPhuI6SDh0azZnXX4uZM01YvgGI=";
    };

    python312 = prev.python312.override {
        packageOverrides = pyfinal: pyprev: {
            autorom = pyfinal.callPackage ./autorom.nix { };

            ale-py = pyprev.ale-py.overrideAttrs (oldAttrs: {
                postFixup = (oldAttrs.postInstall or "") + ''
                    cp ${final.atari-roms}/* $out/${pyprev.python.sitePackages}/ale_py/roms/
                '';
            });
        };
    };
}
