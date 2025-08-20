{
  description = "COE 379L (Fall 2025) Class Jupyter Server";
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
      requests
      seaborn
      scikit-learn
      scikit-image
      scipy
      tensorflow
    ]);
    commonPackages = [ python pkgs.rsync pkgs.gnumake pkgs.lesspipe pkgs.less pkgs.coreutils pkgs.bashInteractive pkgs.which ];
    in {
        devShells.default = pkgs.mkShell {
          name = "379L";
          buildInputs = commonPackages;
          shellHook = ''
           eval "$(lesspipe.sh)"
           '';
        };
      }
    );
}