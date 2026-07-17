#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "rtweekend.h"



// Une couleur EST un vec3 (r,g,b dans [0,1]) — l'alias ne crée pas de nouveau type,
// il documente l'intention. Bonus : toute l'algèbre de vec3 s'applique gratuitement
// aux couleurs (moyenne d'échantillons, atténuation canal par canal via operator*...).
using color = vec3;

// Convertit une couleur en un pixel PPM "P3" : trois entiers 0-255 séparés
// par des espaces, un pixel par ligne. L'en-tête du fichier (P3, dimensions, 255)
// est écrit une seule fois, avant la boucle de rendu.
inline void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // 256 * clamp(x, 0, 0.999) : même idée que l'ancien 255.999 — découper [0,1]
    // en 256 tranches égales avant troncature (int() tronque, n'arrondit pas),
    // le plafond 0.999 garantissant que l'octet ne dépasse jamais 255. Le clamp
    // ajoute la robustesse : une composante hors [0,1] (accumulation d'échantillons,
    // futurs matériaux émissifs) serait sinon convertie en octet invalide.
    static const interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));

    // (À venir dans le livre : correction gamma avant cette conversion — les valeurs
    // linéaires calculées par le rendu ne correspondent pas à la perception de
    // l'écran, il faudra passer par sqrt() ≈ gamma 2.)

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
