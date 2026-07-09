# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Nature du projet

Dépôt d'apprentissage personnel : programmation CUDA (en suivant https://github.com/Infatoshi/cuda-course) et rendu par lancer de rayons (en suivant *Ray Tracing in One Weekend* de Peter Shirley). Le raytracer est pour l'instant en C++ pur sur CPU ; l'objectif à terme est son portage en CUDA (cf. l'article de Roger Allen *Accelerated Ray Tracing in One Weekend in CUDA*).

Le code source est écrit en anglais, mais les commentaires et la documentation sont en **français**. Les commentaires sont volontairement pédagogiques et détaillés (ils expliquent le raisonnement mathématique/géométrique, pas seulement le code) — conserver ce style lors des modifications.

## Compilation et exécution

```sh
cmake -S . -B build        # configuration (une fois)
cmake --build build -j     # compilation
```

Les exécutables sortent dans `build/bin/` :

```sh
./build/bin/idxing                              # exercice d'indexation CUDA
./build/bin/raytracer > output/image.ppm        # rendu PPM sur stdout
```

Le raytracer écrit l'image PPM sur stdout et sa progression sur stderr — toujours rediriger stdout vers un fichier `.ppm`. Pas de tests ni de linter configurés.

Chaque nouvel exécutable s'ajoute par un `add_executable` dans `CMakeLists.txt`. L'architecture CUDA est détectée automatiquement (`CMAKE_CUDA_ARCHITECTURES native`). C++17 et CUDA 17 requis.

## Structure

- `src/01_idxing.cu` — exercices CUDA numérotés (indexation thread/block/grid). Les fichiers `.cu` suivants suivront ce schéma de nommage `NN_sujet.cu`.
- `src/raytracer/` — raytracer C++ CPU : `main.cc` (caméra, viewport, boucle de rendu, intersection rayon-sphère), `vec3.h` (vecteurs 3D, alias `point3` et `color`), `ray.h`, `color.h` (écriture PPM). Suit fidèlement la progression du livre de Shirley.
- `docs/notes.md` — journal de questions/réponses sur les concepts CUDA et le rendu (warps, SM, Monte-Carlo, équation du rendu…). S'y référer pour connaître le niveau et les objectifs de l'apprentissage.
- `docs/recherche_gpu_rendu.md` — document de recherche structuré tiré de ces notes.
- `docs/diagrams/` — schémas SVG accompagnant les notes.
- `output/` et `build/` — non versionnés (`.gitignore`).
