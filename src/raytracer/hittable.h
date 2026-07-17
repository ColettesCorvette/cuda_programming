#ifndef HITTABLE_H
#define HITTABLE_H

#include "interval.h"
#include "rtweekend.h"


// Tout ce qu'il faut savoir d'un impact, indépendamment de la forme touchée :
// le point, la normale, le t du rayon, et de quel côté de la surface on arrive.
class hit_record {
    public:
    point3 p;
    vec3 normal;
    double t;
    bool front_face;   // true : le rayon frappe la surface depuis l'EXTÉRIEUR

    // Convention du livre : la normale stockée pointe TOUJOURS contre le rayon
    // (utile au moment du shading, on la fige ici plutôt qu'à chaque usage).
    // Test : dot(d, n_ext) < 0 <=> le rayon et la normale extérieure se font
    // face <=> on arrive de l'extérieur. Sinon (intérieur d'une sphère de
    // verre, par ex.), on retourne la normale.
    // Prérequis : outward_normal doit être UNITAIRE (c'est la géométrie qui le
    // garantit, cf. sphere.h — la diviser par le rayon suffit).
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }

};

// Interface commune de tout objet "touchable" par un rayon. hit() ne rapporte
// un impact que si son t est DANS ray_t (l'interval remplace l'ancienne paire
// tmin/tmax) — c'est ce paramètre qui permet à hittable_list de rétrécir la
// fenêtre de recherche au fil des objets.
class hittable {
    public:
    virtual ~hittable() = default;

    virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif
