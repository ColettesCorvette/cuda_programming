#include <cmath>
#include <iostream>
#include "color.h"
#include "vec3.h"
#include "ray.h"


// Un point P est sur la sphère <=> (C-P)·(C-P) = r² (distance au centre = r,
// élevée au carré pour éviter une racine). En remplaçant P par P(t) = origine + t*direction
// (l'équation du rayon), développer ce produit scalaire donne un polynôme du 2nd degré en t :
// a*t² + b*t + c = 0, avec a = d·d, b = -2*d·(C-Q), c = (C-Q)·(C-Q) - r².
//
// Le discriminant compte les intersections réelles :
//   < 0 : le rayon rate la sphère (aucun t ne met le point pile sur la surface)
//   = 0 : le rayon est tangent (un seul point de contact)
//   > 0 : le rayon entre et ressort (deux points, t1 < t2)
// (on ne garde pour l'instant que l'existence d'une solution, pas la plus petite racine
// positive qui donnerait le premier impact)
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = h*h - a*c;
    
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - std::sqrt(discriminant) ) / a;
    }
}

color ray_color(const ray& r) {
    auto t = hit_sphere((point3(0,0,-1)), 0.5, r);
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        return 0.5*color(N.x()+1,N.y()+1, N.z()+1);
    }
        
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

int main() {
    

    //Image
    // aspect_ratio = largeur/hauteur, par définition -> hauteur = largeur/aspect_ratio
    // (pure algèbre, pas de magie : c'est juste l'équation de définition réarrangée).
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    //Camera
    // Viewport widths less than one are ok since they are real valued.
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Le viewport est un rectangle virtuel DANS LA SCÈNE 3D (pas l'image des pixels) :
    // c'est la "fenêtre" à travers laquelle on lance les rayons. Vu va du bord gauche
    // au bord droit ; Vv va du bord HAUT vers le BAS -> le -viewport_height absorbe
    // l'inversion d'axe entre l'image (j augmente en descendant) et le monde 3D
    // (y augmente en montant). Grâce à ce signe, i et j croissants avancent naturellement
    // dans le sens de lecture de l'image, sans plus jamais se soucier de l'inversion.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Delta = taille d'une cellule de pixel = aussi l'écart entre deux CENTRES voisins
    // (les demi-cellules de part et d'autre s'annulent quand on va d'un centre au suivant).
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Construction de Q (coin supérieur gauche du viewport), en partant du centre caméra :
    //  1. -(0,0,focal_length) : avance vers la scène (la caméra regarde vers -z) -> centre du viewport
    //  2. -viewport_u/2       : recule d'une demi-largeur vers la gauche -> milieu du bord gauche
    //  3. -viewport_v/2       : Vv pointe vers le bas, donc EN SOUSTRAIRE remonte -> coin haut-gauche
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    // P00 = Q + une demi-cellule en largeur ET en hauteur, pour centrer le premier pixel
    // dans sa cellule plutôt que de le coller au coin du viewport (0.5*(du+dv) = du/2 + dv/2).
    // Ensuite : position(i,j) = P00 + i*pixel_delta_u + j*pixel_delta_v.
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);




    //Render
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // Boucle externe j = ligne courante (haut -> bas), boucle interne i = pixels de
    // cette ligne (gauche -> droite). i "tourne vite" (0..image_width-1) pendant que j
    // reste fixe, puis j avance d'un cran : on écrit bien ligne par ligne, comme
    // l'attend le format PPM.
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {

            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = ray_color(r);
            write_color(std::cout, pixel_color);
            
        }
    }
    std::clog << "\rDone.                 \n";
}