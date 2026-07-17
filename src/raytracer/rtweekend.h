#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

//C++ std

using std::make_shared;
using std::shared_ptr;

//Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

//Utility Functions

inline double degrees_to_radian(double degrees) {
    return degrees * pi / 180;
}

// Réel aléatoire dans [0,1) — la source d'aléa de tout le Monte-Carlo à venir.
// static : UN générateur global, initialisé au premier appel puis réutilisé
// (graine par défaut fixe -> rendu reproductible d'une exécution à l'autre).
// Mersenne Twister plutôt que rand() : distribution de bien meilleure qualité.
// (Au portage CUDA : à remplacer par cuRAND, un état de générateur PAR THREAD —
// un état global serait un point de contention pour des milliers de threads.)
inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

// Réel aléatoire dans [min,max) : simple remise à l'échelle du [0,1) ci-dessus.
inline double random_double(double min, double max) {
    return min + (max-min)*random_double();
}

//Common Headers

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "interval.h"


#endif
