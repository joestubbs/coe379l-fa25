{
  description = "COE 379L (Fall 2025) Documentation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        docsPython = python.withPackages (ps: [
          ps.sphinx
          ps.sphinx-autobuild
          ps.sphinx_rtd_theme
          ps.sphinx-tabs
          ps.docutils
          ps.setuptools
        ]);
        commonPackages = [ docsPython pkgs.rsync pkgs.gnumake ];
      in {
        devShells = {
          default = pkgs.mkShell {
            packages = commonPackages;
          };
        };
        packages = {
          default = pkgs.stdenv.mkDerivation {
            # builds to ./result
            name = "coe379l-docs";
            src = ./.;
            buildInputs = commonPackages;
            buildPhase = ''
              make html
            '';
            installPhase = ''
              mkdir -p $out
              cp -r build/html $out/
            '';
          };
        };
      }
    );
}