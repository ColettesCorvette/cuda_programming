#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

using color = vec3;

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

    // Write out the pixel color components.
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif

