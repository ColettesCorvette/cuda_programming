#ifndef SPHERE_H
#define SPHERE_H

#include "rtweekend.h"
#include "hittable.h"


class sphere : public hittable {
  public:
    // fmax(0, radius) : un rayon négatif n'a pas de sens géométrique — on le
    // neutralise à la construction plutôt que de le tester à chaque hit.
    sphere(const point3& center, double radius) : center(center), radius(std::fmax(0,radius)) {}

    // Un point P est sur la sphère <=> (C-P)·(C-P) = r² (distance au centre = r,
    // au carré pour éviter une racine). En remplaçant P par P(t) = Q + t*d
    // (l'équation du rayon), développer ce produit scalaire donne un polynôme du
    // 2nd degré en t : a*t² + b*t + c = 0, avec a = d·d, b = -2*d·(C-Q),
    // c = (C-Q)·(C-Q) - r².
    //
    // Simplification : b est un multiple de -2 (le terme croisé du carré sort
    // toujours avec son facteur 2), donc on pose h = -b/2 = d·(C-Q). La formule
    // des racines (-b ± √(b²-4ac)) / 2a devient (h ± √(h²-ac)) / a : les
    // facteurs 2 et 4 s'annulent — mêmes racines, moins d'opérations. h²-ac
    // joue le rôle du discriminant (même signe que b²-4ac).
    //
    // Le discriminant compte les intersections réelles :
    //   < 0 : le rayon rate la sphère
    //   = 0 : tangent (un point de contact)
    //   > 0 : le rayon entre et ressort (deux racines, t1 < t2)
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // La plus proche racine d'abord (h - √disc : le premier impact le long
        // du rayon) ; si elle tombe hors de ray_t, on tente la lointaine —
        // cas où l'origine du rayon est DANS la sphère : l'entrée est derrière
        // nous (t < 0), seule la sortie compte.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        // (P - C)/r : normale extérieure, unitaire PAR CONSTRUCTION (|P-C| = r,
        // pas besoin de normaliser) — c'est le prérequis de set_face_normal,
        // qui l'oriente ensuite contre le rayon et remplit front_face.
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }

  private:
    point3 center;
    double radius;
};

#endif
