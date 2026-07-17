#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"


int main() {

    // World : la scène est une hittable_list (polymorphisme : tout objet "touchable").
    // Deux sphères — la petite est le sujet ; la géante (r=100, centrée 100.5 unités
    // sous l'axe) sert de sol : son sommet affleure y=-0.5, et à cette échelle sa
    // courbure est localement imperceptible. Astuce du livre pour avoir un "plan"
    // sans écrire une nouvelle primitive.
    hittable_list world;

    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    // Camera : toute la géométrie du rendu (viewport, deltas de pixel, P00, boucle
    // de scanlines) vit désormais dans camera.h — main() ne fait que régler les
    // paramètres "utilisateur", render() dérive le reste dans initialize().
    camera cam;

    cam.aspect_ratio = 16.0/9.0;

    cam.image_width = 400;
    cam.samples_per_pixel = 100;   // antialiasing : moyenne de 100 rayons par pixel

    cam.render(world);

}
