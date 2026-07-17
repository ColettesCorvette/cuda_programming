#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include <vector>
#include "interval.h"
#include "rtweekend.h"

// La scène : une liste d'objets, elle-même hittable (composite). shared_ptr car
// plusieurs listes pourront partager les mêmes objets (et la durée de vie est
// gérée automatiquement — point qui sautera au portage CUDA).
class hittable_list: public hittable {
    public:
        std::vector<shared_ptr<hittable>> objects;

        hittable_list() {}
        hittable_list(shared_ptr<hittable> object) { add(object); }

        void clear() {objects.clear();}

        void add (shared_ptr<hittable> object) {
            objects.push_back(object);
        }

        // Recherche de l'impact le plus proche : l'intervalle se RÉTRÉCIT en
        // marchant la liste. Chaque hit abaisse closest_so_far, et l'objet
        // suivant est interrogé sur [min, closest_so_far] : tout impact plus
        // lointain que le meilleur courant est rejeté par l'objet lui-même,
        // sans comparaison ici. Même logique "t minimal" que la petite racine
        // de sphere::hit, un étage au-dessus. (La BVH remplacera un jour ce
        // parcours linéaire O(n) — cf. docs/recherche_gpu_rendu.md, section 6.)
        bool hit(const ray& r, interval ray_t, hit_record& rec ) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for(const auto& object : objects) {
                if(object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            return hit_anything;
        }
};

#endif
