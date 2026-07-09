# Design — Document de recherche GPU + schéma anatomique + commentaires

Date : 2026-07-09
Statut : décisions prises en mode autonome, validées implicitement par déblocage des permissions.

## Objectifs

1. Améliorer les commentaires des fichiers sources.
2. Transformer `docs/notes.md` + les 6 diagrammes en un document de recherche personnel structuré.
3. Créer un schéma SVG annoté : GPU relié au PC par ses bus, avec zooms successifs jusqu'au silicium.

## Décisions

### Document de recherche — `docs/recherche_gpu_rendu.md`

- **Format** : Markdown versionné (lisible sur GitHub, éditable, diffable). `notes.md` reste intact comme journal brut.
- **Structure** : thématique et progressive, chaque section ouvre sur la question d'origine (citation) puis la synthèse. Sections :
  1. Le modèle d'exécution CUDA — grid/block logiques vs SM/warp matériels
  2. Le warp — unité d'exécution réelle (SIMT, divergence, coalescence)
  3. Anatomie du GPU — CUDA cores, SM, hiérarchie mémoire
  4. Où CUDA est utilisé — écosystème, DirectX, compute shaders
  5. RT Cores et Tensor Cores
  6. Ray tracing, path tracing et Monte-Carlo — équation du rendu, N dimensions, 1/√N
  7. Simuler Maxwell ? — FDTD et limite asymptotique
  8. Feuille de route — RTiOW C++ → portage CUDA → OptiX
- SVG existants intégrés par référence (`![](diagrams/…)`) à l'endroit pertinent, nouveau schéma inclus.

### Schéma anatomique — `docs/diagrams/gpu_anatomie_bus.svg`

Un seul SVG vertical à 5 panneaux (niveaux de zoom), reliés par des flèches « loupe » :

- **Niveau 1 — Le système** : CPU (host), RAM, GPU (device) reliés par PCIe x16 ; annotations `cudaMemcpy` H2D/D2H, lancement de kernel, débits (PCIe 4.0 ~32 Go/s).
- **Niveau 2 — La carte** : PCB avec die GPU, puces GDDR autour, bus mémoire 384-bit ~1 To/s, étage d'alimentation, connecteur PCIe.
- **Niveau 3 — Le die** : GigaThread Engine, rangées de SM, L2 partagé, contrôleurs mémoire ; annotation « grid = file de blocks ».
- **Niveau 4 — Le SM** : warp schedulers ×4, register file 256 Ko, 128 ALU FP32, Tensor/RT cores, SFU, LD/ST, shared memory/L1 128 Ko ; annotations block→SM, warp→scheduler, thread→lane.
- **Niveau 5 — Le silicium** : une ALU = additionneur/multiplicateur FMA → porte logique CMOS → transistor (FinFET ~5 nm).
- Style : palette et typographie des 6 SVG existants (fonds beige `#F5F4ED`/`#F1EFE8`, vert `#0F6E56` pour le calcul, texte `Anthropic Sans`).

### Commentaires sources

- `src/01_idxing.cu` : commentaires anglais (analogie « apartment ») réécrits en français pédagogique, cohérents avec le style du raytracer ; ajout : ordre row-major, lien avec les warps (64 threads = 2 warps), rôle de `cudaDeviceSynchronize` (lancement asynchrone, flush du printf device).
- `src/raytracer/color.h` : expliquer le rôle du fichier, le lien couleur=vec3, la troncature 255.999 (déjà fait), le futur gamma.
- `src/raytracer/main.cc` : clarifier la simplification h = −b/2 dans `hit_sphere` (le commentaire actuel donne le b classique mais le code utilise h).
- `vec3.h`, `ray.h` : déjà au bon niveau, pas de changement significatif.

## Hors périmètre

- Pas de portage CUDA du raytracer (étape future).
- Pas de commit automatique (à la demande de l'utilisateur uniquement).
- `notes.md` non modifié.
