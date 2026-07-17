#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "interval.h"
#include "rtweekend.h"

// La caméra concentre tout ce qui était éparpillé dans main() : la géométrie du
// viewport, la génération des rayons et la boucle de rendu. Deux zones :
//  - public  : les paramètres que l'utilisateur règle AVANT d'appeler render() ;
//  - private : les grandeurs dérivées, calculées par initialize() au début du rendu.
class camera {
    public:

        double aspect_ratio = 1;        // largeur/hauteur demandée
        int image_width = 100;          // largeur de l'image en pixels
        int samples_per_pixel = 10;     // rayons lancés par pixel (antialiasing)

        void render(const hittable& world) {
            initialize();

            //Render
            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            // Boucle externe j = ligne courante (haut -> bas), boucle interne i = pixels de
            // cette ligne (gauche -> droite). i "tourne vite" (0..image_width-1) pendant que j
            // reste fixe, puis j avance d'un cran : on écrit bien ligne par ligne, comme
            // l'attend le format PPM.
            for (int j = 0; j < image_height; j++) {
                std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
                for (int i = 0; i < image_width; i++) {
                    // Premier estimateur Monte-Carlo du projet : la couleur du pixel
                    // = moyenne de N rayons tirés au hasard DANS la cellule du pixel
                    // (au lieu d'un seul rayon au centre). Les bords d'objets, qui ne
                    // couvrent qu'une fraction de la cellule, reçoivent une couleur
                    // proportionnelle à cette fraction -> plus d'escaliers. Le bruit
                    // résiduel décroît en 1/sqrt(N).
                    color pixel_color(0,0,0);
                    for (int sample = 0; sample < samples_per_pixel; sample++) {
                        ray r = get_ray(i, j);
                        pixel_color += ray_color(r, world);
                    }
                    write_color(std::cout, pixel_samples_scale * pixel_color);
                }
            }
            std::clog << "\rDone.                 \n";
        }

    private:
        int    image_height;          // hauteur en pixels, dérivée de width/ratio
        point3 center;                // centre de la caméra (origine des rayons)
        point3 pixel00_loc;           // centre du pixel (0,0)
        vec3   pixel_delta_u;         // pas d'un centre de pixel au suivant, en largeur
        vec3   pixel_delta_v;         // idem en hauteur (pointe vers le BAS)
        double pixel_samples_scale;   // 1/N, précalculé : la division de la moyenne

        void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = point3(0, 0, 0);

            auto focal_length = 1.0;
            auto viewport_height = 2.0;
            auto viewport_width = viewport_height * (double(image_width)/image_height);

            // Le viewport est un rectangle virtuel DANS LA SCÈNE 3D (pas l'image des
            // pixels) : la "fenêtre" à travers laquelle on lance les rayons. Vu va du
            // bord gauche au bord droit ; Vv va du bord HAUT vers le BAS -> le
            // -viewport_height absorbe l'inversion d'axe entre l'image (j augmente en
            // descendant) et le monde 3D (y augmente en montant). Grâce à ce signe,
            // i et j croissants avancent naturellement dans le sens de lecture de
            // l'image, sans plus jamais se soucier de l'inversion.
            auto viewport_u = vec3(viewport_width, 0, 0);
            auto viewport_v = vec3(0, -viewport_height, 0);

            // Delta = taille d'une cellule de pixel = aussi l'écart entre deux CENTRES
            // voisins (les demi-cellules de part et d'autre s'annulent d'un centre au suivant).
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Construction de Q (coin supérieur gauche du viewport), depuis le centre caméra :
            //  1. -(0,0,focal_length) : avance vers la scène (la caméra regarde vers -z)
            //  2. -viewport_u/2       : recule d'une demi-largeur -> milieu du bord gauche
            //  3. -viewport_v/2       : Vv pointe vers le bas, donc EN SOUSTRAIRE remonte
            auto viewport_upper_left =
            center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
            // P00 = Q + une demi-cellule en largeur ET en hauteur : le premier pixel est
            // CENTRÉ dans sa cellule, pas collé au coin (0.5*(du+dv) = du/2 + dv/2).
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        }

        // Fabrique le rayon du sample courant : au lieu de viser le centre exact du
        // pixel (i,j), on vise un point aléatoire de sa cellule. offset dans
        // [-0.5,+0.5]² s'ajoute aux indices AVANT la multiplication par les deltas :
        // on reste donc dans la cellule, par construction.
        ray get_ray(int i, int j) const {
            auto offset = sample_square();
            auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

            auto ray_origin = center;
            auto ray_direction = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        // Point aléatoire du carré unité centré [-0.5,+0.5]², z=0 (le viewport est
        // un plan : l'aléa n'a que deux dimensions).
        vec3 sample_square() const {
            return vec3(random_double() - 0.5, random_double() - 0.5, 0);
        }

        color ray_color(const ray& r, const hittable& world) {
            hit_record rec;
            // interval(0, infinity) : on n'accepte que les impacts DEVANT la caméra
            // (t > 0) ; le monde entier décide du plus proche (cf. hittable_list).
            if(world.hit(r, interval(0, infinity), rec)) {
                // Visualisation de la normale : ses composantes sont dans [-1,1],
                // le +1 puis *0.5 les remappe en [0,1] -> une couleur valide.
                return 0.5 * (rec.normal + color(1,1,1));
            }
            // Aucun impact : dégradé de ciel. a dans [0,1] selon la hauteur du rayon,
            // puis interpolation linéaire (lerp) blanc -> bleu.
            vec3 unit_direction = unit_vector(r.direction());
            auto a = 0.5*(unit_direction.y() + 1.0);
            return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
        }
};
#endif
