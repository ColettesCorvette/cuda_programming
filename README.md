# cuda_programming

## Prérequis

- CMake >= 3.18
- CUDA Toolkit (nvcc)
- Un compilateur C++ (g++)

## Compilation

```sh
cmake -S . -B build
cmake --build build -j
```

Les exécutables sont générés dans `build/bin/` :

```sh
./build/bin/idxing
./build/bin/raytracer > output/image.ppm
```
