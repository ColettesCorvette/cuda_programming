#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    // Constructeur par défaut : permet de déclarer un ray avant de le remplir
    // (ex: tableau de rayons, membre ray initialisé plus tard). Comme la classe
    // définit aussi un constructeur à 2 paramètres, celui-ci ne serait pas généré
    // automatiquement — il faut le redéclarer explicitement pour l'avoir.
    ray() {}

    // origin_(origin), direction_(direction) : liste d'initialisation des membres.
    // Construit directement origin_/direction_ avec la bonne valeur (via le
    // constructeur de copie implicite de vec3), au lieu de les construire par
    // défaut puis de les réaffecter dans le corps {} (une étape en moins).
    // origin_/direction_ (suffixe _) évite toute confusion avec les paramètres
    // origin/direction et avec les getters origin()/direction() ci-dessous.
    ray(const point3& origin, const vec3& direction) : origin_(origin), direction_(direction) {}

    // const point3& (retour) : référence constante, pas de copie du membre, et
    // l'appelant ne peut pas modifier origin_ à travers cette référence.
    // const (méthode) : ne modifie pas l'objet, donc appelable sur un ray reçu
    // en const ray&.
    const point3& origin() const  { return origin_; }
    const vec3& direction() const { return direction_; }

    point3 at(double t) const {
        return origin_ + t*direction_;
    }

  private:
    point3 origin_;
    vec3 direction_;
};

#endif