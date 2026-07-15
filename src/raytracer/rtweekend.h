#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

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

//Common Headers 

#include "vec3.h"
#include "color.h"
#include "ray.h"


#endif