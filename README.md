# Path tracer implementation & cuda_programming
![Platform](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)


## Prérequis

- CMake >= 3.18
- CUDA Toolkit (nvcc)
- Un compilateur C++ (g++)

## Lint 

```sh
clang-tidy src/main.cpp        # trouve compile_commands.json tout seul
run-clang-tidy -p build        # tout le projet en parallèle
clang-tidy --fix src/main.cpp  # applique les corrections automatiques
```

## Compilation

GCC
```sh
cmake -S . -B build
cmake --build build -j
```

Avec llvm

```sh
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug #RelWithDebInfo
cmake --build build
```
## Execution

Les exécutables sont générés dans `build/bin/` :

```sh
./build/bin/idxing
./build/bin/raytracer > output/image.ppm
```
gdb : 

## Debug

```sh
gdb ./build/bin/
```

commandes :

```sh
b main.cpp:42      # breakpoint ligne 42
b World::step      # breakpoint sur une fonction
r                  # run
n / s              # next (ligne suivante) / step (entre dans la fonction)
c                  # continue jusqu'au prochain breakpoint
p body.velocity    # affiche une variable (comprend les objets C++)
bt                 # backtrace — où suis-je et par où suis-je passé
watch x            # s'arrête quand x change de valeur
f 2                # saute à la frame 2 de la backtrace
q                  # quitter
```

## Profiler 

Build en RelWithDebInfo, puis :

```sh
perf stat ./build/bin/            # vue d'ensemble : cycles, cache misses, IPC
perf record -g ./build/bin/        # échantillonne avec les call graphs
perf report                        # explorateur interactif dans le terminal
```

perf stat seul est déjà très instructif pour toi : il te montre l'IPC (instructions par cycle) et les cache misses — exactement les métriques qui comptent quand tu itères sur un tableau de particules et que tu te demandes si ton layout mémoire (AoS vs SoA) est bon.
Pour la visualisation, Hotspot (pacman -S hotspot) est une GUI qui ouvre les fichiers perf.data et génère les flamegraphs directement — plus simple que les scripts FlameGraph de Brendan Gregg pour commencer. Un flamegraph te montre en un coup d'œil quelle fonction mange ton temps CPU : largeur = temps, hauteur = profondeur de pile.
Complément : valgrind --tool=callgrind ./build/sim puis kcachegrind pour visualiser. Plus précis que perf (comptage exact, pas échantillonnage) mais 20-50x plus lent — utilisable sur des petits runs seulement.


## Exemple de workflow 

```sh
# une fois par projet
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
ln -s build/compile_commands.json .

# boucle de dev
cmake --build build && ./build/bin/   # sanitizers actifs
# clangd te montre les warnings clang-tidy en live dans Helix

# ça crash bizarrement → gdb ./build/sim
# c'est lent → rebuild en RelWithDebInfo → perf record -g → hotspot
```
