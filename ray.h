#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    ray() {}

    ray(const point3& origin, const vec3& direction) : origin_(origin), direction_(direction) {}

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