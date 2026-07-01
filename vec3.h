#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {
  public:
    double e[3];

    // Le tableau e est initialisé via la liste d'initialisation (après ':');
    // le corps {} est donc vide, mais reste obligatoire pour définir la fonction.
    vec3() : e{0,0,0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    // const : ces méthodes ne modifient pas l'objet, donc utilisables sur un vec3 const
    // (ex: quand le vecteur est reçu en paramètre par const vec3&).
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    // Deux versions de operator[] : celle-ci (const, retourne une valeur) sert à LIRE,
    // ex: v[0] sur un vecteur qu'on ne modifie pas.
    double operator[](int i) const { return e[i]; }
    // Celle-ci retourne une référence (double&) vers la vraie case du tableau,
    // ce qui permet d'ÉCRIRE : v[0] = 5.0 modifie réellement e[0].
    double& operator[](int i) { return e[i]; }

    // Modifient l'objet sur place (comme +=, *=, /= sur des nombres), composante par
    // composante. const vec3& v en paramètre = pas de copie de v, et on ne le modifie pas.
    // Retourne vec3& (référence vers *this, pas une copie) pour permettre le chaînage.
    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    // Réutilise *= plutôt que de refaire le calcul (diviser par t = multiplier par 1/t).
    vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    double length() const {
        return std::sqrt(length_squared());
    }

    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
//
// Fonctions libres (hors classe) car soit l'opérande de gauche n'est pas un vec3
// (ex: double * vec3, ou std::ostream << vec3), soit l'opération est symétrique
// (a + b) et crée un NOUVEAU vecteur sans modifier a ni b (contrairement à +=).
//
// inline : nécessaire car ce header peut être inclus dans plusieurs .cpp. Sans inline,
// chaque .cpp aurait sa propre définition et l'éditeur de liens rejetterait les doublons
// (violation de l'ODR). Les méthodes définies dans la classe sont, elles, implicitement
// inline.

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// Produit composante par composante (PAS le produit scalaire ni vectoriel).
// Sert surtout pour les couleurs (vec3 réutilisé comme color) : multiplier une
// lumière par une surface canal par canal (r*r, g*g, b*b).
inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

// Produit scalaire : retourne un double, utile pour les angles/projections.
// Si u et v sont unitaires, dot(u,v) = cos(angle) entre eux (ex: éclairage diffus,
// loi de Lambert avec la normale et la direction de la lumière).
inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

// Produit vectoriel : retourne un vec3 perpendiculaire à u et v (ex: construire un repère).
inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

// Normalise v (même direction, longueur 1). Indispensable dès qu'on ne veut que la
// direction pure sans que la longueur ne fausse les calculs (directions de rayons,
// normales de surface avant un dot()).
inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif