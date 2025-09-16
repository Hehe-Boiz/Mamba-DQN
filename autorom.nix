{
    pkgs,
    lib,
    buildPythonPackage,
    fetchFromGitHub,
    setuptools,
    wheel,
    click,
    requests,
    farama-notifications
}:
buildPythonPackage rec {
    pname = "autorom";
    version = "0.6.1";

    propagatedBuildInputs = [ 
        click
        requests
        farama-notifications
    ];

    src = fetchFromGitHub {
        owner = "Farama-Foundation";
        repo = "AutoROM";
        rev = "v${version}";
        sha256 = "sha256-fC5OOXAnnP4x4j/IbpG0YdTz5F5pgyY0tumNjyrQ8FM=";
    };

    sourceRoot = "source/packages/AutoROM";

    postPatch = ''
        sed -i '1i import tempfile' src/AutoROM.py

        substituteInPlace src/AutoROM.py \
            --replace "os.path.dirname(__file__)" \
                "tempfile.gettempdir()"
    '';

    doCheck = false;

    pyproject = true;
    build-system = [
        setuptools
        wheel
    ];
}
