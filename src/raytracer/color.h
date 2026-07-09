#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

// Une couleur EST un vec3 (r,g,b dans [0,1]) — l'alias ne crée pas de nouveau type,
// il documente l'intention. Bonus : toute l'algèbre de vec3 s'applique gratuitement
// aux couleurs (moyenne d'échantillons, atténuation canal par canal via operator*...).
using color = vec3;

// Convertit une couleur [0,1]³ en un pixel PPM "P3" : trois entiers 0-255 séparés
// par des espaces, un pixel par ligne. L'en-tête du fichier (P3, dimensions, 255)
// est écrit une seule fois par main(), avant la boucle de rendu.
void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    // 255.999 (pas 255) : découpe [0,1] en 256 tranches égales avant troncature (int()
    // tronque, ne fait pas d'arrondi). Avec 255, seule la valeur EXACTE r=1.0 donnerait
    // 255 : le blanc pur serait quasi inatteignable et les tranches seraient décalées.
    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    // (À venir dans le livre : correction gamma avant cette conversion — les valeurs
    // linéaires calculées par le rendu ne correspondent pas à la perception de
    // l'écran, il faudra passer par sqrt() ≈ gamma 2.)

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
