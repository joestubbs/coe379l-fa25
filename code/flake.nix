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
    commonPackages = [ python pkgs.rsync pkgs.gnumake pkgs.lesspipe pkgs.less pkgs.coreutils pkgs.bashInteractive pkgs.which ];

    in rec {
        devShell = shell {
          name = "379L";
          buildInputs = commonPackages;
        };

 #       dockerImage = pkgs.dockerTools.buildLayeredImage {
 #           name = "jstubbs/coe79L";
 #           tag = "fa25";
 #           contents = [ commonPackages ];
 #           created = "now"; # Fix create time in docker images
 #           config = {
 #               Cmd = [ "${devShell}/bin/bash" ]; # Set the entrypoint to a shell
 #               #Cmd = [ "${pkgs.bash}/bin/bash" ]; # Set the entrypoint to a shell
 #               WorkingDir = "/coe379L"; # Set a working directory
 #           };
 #       };
        dockerImage = pkgs.dockerTools.buildNixShellImage {
            name = "jstubbs/coe79L";
            tag = "fa25";
            drv = devShell;
        };

        packages = {
            docker = dockerImage;
            default = dockerImage;
        };
      });
}