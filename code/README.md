# Software for COE 332

This directory contains software and other code snippets that 
can be used for the class. Each subsection below describes the 
individual software components. 

## Notebook Server 
One software component available here is the Jupyter notebook server used for the class.


1. Build the docker image using nix

```
 nix build .#packages.x86_64-linux.docker
```

2. Load the tar.gz as a tagged docker image

```
    docker load < result
```

3. Run the image locally:

```
docker run -p 8888:8888 -it --rm jstubbs/coe79l:fa25
```
By default, this starts you up in the custom Nix development shell. From there 
you can start various commands, e.g., Jupyter:

```
[nix 379L] ~ ➜ jupyter-notebook --ip 0.0.0.0
```