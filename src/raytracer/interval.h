#ifndef INTERVAL_H
#define INTERVAL_H

#include "rtweekend.h"

// Un intervalle réel [min, max]. Remplace les paires (tmin, tmax) qui se
// propageaient dans toutes les signatures de hit() : une seule valeur à passer,
// et le vocabulaire (contains, surrounds, clamp) dit ce qu'on teste au lieu de
// réécrire des comparaisons à la main.
class interval {
  public:
    double min, max;

    // Par défaut : intervalle VIDE. min=+inf > max=-inf, donc aucun x ne peut
    // satisfaire min <= x <= max — c'est le vide exprimé sans cas particulier.
    interval() : min(+infinity), max(-infinity) {}

    interval(double min, double max) : min(min), max(max) {}

    double size() const {
        return max - min;
    }

    // contains : bornes INCLUSES (<=). surrounds : bornes EXCLUES (<) — c'est lui
    // qu'utilise sphere::hit, pour rejeter un impact pile à t=tmin ou t=tmax
    // (typiquement t=0 : l'origine du rayon elle-même n'est pas un impact valable).
    bool contains(double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }

    // Ramène x dans l'intervalle (utilisé par write_color pour borner les
    // couleurs avant conversion en octets).
    double clamp(double x) const {
        if(x < min) return min;
        if(x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif
