#include <stdio.h>

// Chaque thread calcule son identifiant GLOBAL unique à partir de ses coordonnées
// 3D dans la grid (blockIdx) et dans son block (threadIdx). C'est la linéarisation
// classique "row-major" : x tourne le plus vite, puis y, puis z — exactement comme
// on numérote les cases d'un tableau 3D aplati en mémoire.
//
// Analogie : la grid est une ville de gridDim.z quartiers, contenant chacun
// gridDim.y rues de gridDim.x immeubles ; chaque immeuble (block) héberge
// blockDim.x*y*z habitants (threads). L'id global = numéro de l'habitant dans
// la ville entière.
__global__ void whoami(void) {
    // Numéro linéaire du block dans la grid (même formule que pour un tableau 3D) :
    //   x = position dans la rangée, y saute des rangées entières (gridDim.x blocks
    //   chacune), z saute des "étages" entiers de grid (gridDim.x * gridDim.y blocks).
    int block_id =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Nombre de threads contenus dans tous les blocks qui précèdent le nôtre :
    // c'est l'offset de départ de notre block dans la numérotation globale.
    int block_offset =
        block_id *
        blockDim.x * blockDim.y * blockDim.z; // threads par block

    // Même linéarisation row-major, mais à l'intérieur du block cette fois,
    // avec les dimensions du block (blockDim) au lieu de celles de la grid.
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // id global unique = début du block + position dans le block.
    // C'est LA formule qu'on retrouve partout (en 1D elle se réduit à
    // blockIdx.x * blockDim.x + threadIdx.x, cf. vectorAdd).
    int id = block_offset + thread_offset;

    // printf device : chaque thread écrit dans un buffer sur le GPU, vidé vers
    // le terminal seulement à la synchronisation. L'ordre d'affichage est
    // arbitraire : il reflète l'ordonnancement réel des warps, PAS l'ordre des id
    // — première manifestation visible du parallélisme.
    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4;    // grid  : 2*3*4 = 24 blocks
    const int t_x = 4, t_y = 4, t_z = 4;    // block : 4*4*4 = 64 threads

    // 64 threads/block = 2 warps de 32 : le matériel découpe le block en paquets
    // de 32 threads CONSÉCUTIFS (selon l'id linéaire thread_offset ci-dessus).
    // C'est pourquoi blockDim devrait toujours être un multiple de 32 : un block
    // de 65 threads coûterait 3 warps, le dernier n'ayant qu'1 thread actif sur 32.

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    // dim3 : les dimensions sont purement LOGIQUES (confort d'adressage pour des
    // données 1D/2D/3D). Le GPU, lui, ne voit passer que des blocks à placer sur
    // ses SM et des warps à ordonnancer.
    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();

    // Le lancement de kernel est ASYNCHRONE : le CPU continue immédiatement.
    // Sans cette synchronisation, main() se terminerait avant la fin du kernel
    // et le buffer des printf device ne serait jamais vidé — aucune sortie.
    cudaDeviceSynchronize();
}
