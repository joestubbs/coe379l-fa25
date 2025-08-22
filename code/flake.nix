{
  description = "COE 379L (Fall 2025) Student Jupyter Server";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
    shell-utils.url = "github:waltermoreira/shell-utils";
  };
  outputs = { self, nixpkgs, flake-utils, shell-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      shell = shell-utils.myShell.${system};
      python = pkgs.python312.withPackages(ps: with ps; [
      ipython
      jupyter
      jupyterlab
      keras
      matplotlib
      numpy
      pandas
      pillow
      protobuf
      requests
      seaborn
      scikit-learn
      scikit-image
      scipy
      tensorflow
    ]);
    commonPackages = [ 
      python 
      pkgs.rsync 
      pkgs.gnumake 
      pkgs.lesspipe
      pkgs.less 
      pkgs.coreutils-full
      pkgs.bashInteractive 
      pkgs.which 
      pkgs.file
      pkgs.procps

    ];

    in rec {
        devShell = shell {
          name = "379L";
          buildInputs = commonPackages;
        };

        dockerImage = pkgs.dockerTools.buildNixShellImage {
            name = "jstubbs/coe379l";
            tag = "fa25";
            drv = devShell;
#            run = "sleep infinity";
        };

        packages = {
            docker = dockerImage;
            default = dockerImage;
        };
      });
}