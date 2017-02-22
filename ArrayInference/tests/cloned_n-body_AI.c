// Automatically modified code.
/*
 * The Great Computer Language Shootout
 * http://shootout.alioth.debian.org/
 *
 * contributed by Christoph Bauer
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define pi 3.141592653589793
#define solar_mass (4 * pi * pi)
#define days_per_year 365.24

struct planet {
  double x, y, z;
  double vx, vy, vz;
  double mass;
};

// Generated clone.
void GPU__main__advance(int nbodies, struct planet *bodies, double dt) {
  int i, j;

  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    for (j = i + 1; j < nbodies; j++) {
      struct planet *b2 = &(bodies[j]);
      double dx = b->x - b2->x;
      double dy = b->y - b2->y;
      double dz = b->z - b2->z;
      double distance = sqrt(dx * dx + dy * dy + dz * dz);
      double mag = dt / (distance * distance * distance);
      b->vx -= dx * b2->mass * mag;
      b->vy -= dy * b2->mass * mag;
      b->vz -= dz * b2->mass * mag;
      b2->vx += dx * b->mass * mag;
      b2->vy += dy * b->mass * mag;
      b2->vz += dz * b->mass * mag;
    }
  }
  long long int AI1[28];
  AI1[0] = nbodies + -1;
  AI1[1] = 56 * AI1[0];
  AI1[2] = 16 + AI1[1];
  AI1[3] = 40 + AI1[1];
  AI1[4] = 8 + AI1[1];
  AI1[5] = 32 + AI1[1];
  AI1[6] = 24 + AI1[1];
  AI1[7] = AI1[1] > AI1[6];
  AI1[8] = (AI1[7] ? AI1[1] : AI1[6]);
  AI1[9] = AI1[1] > AI1[8];
  AI1[10] = (AI1[9] ? AI1[1] : AI1[8]);
  AI1[11] = AI1[5] > AI1[10];
  AI1[12] = (AI1[11] ? AI1[5] : AI1[10]);
  AI1[13] = AI1[4] > AI1[12];
  AI1[14] = (AI1[13] ? AI1[4] : AI1[12]);
  AI1[15] = AI1[4] > AI1[14];
  AI1[16] = (AI1[15] ? AI1[4] : AI1[14]);
  AI1[17] = AI1[3] > AI1[16];
  AI1[18] = (AI1[17] ? AI1[3] : AI1[16]);
  AI1[19] = AI1[2] > AI1[18];
  AI1[20] = (AI1[19] ? AI1[2] : AI1[18]);
  AI1[21] = AI1[2] > AI1[20];
  AI1[22] = (AI1[21] ? AI1[2] : AI1[20]);
  AI1[23] = AI1[22] + 56;
  AI1[24] = AI1[23] / 56;
  AI1[25] = (AI1[24] > 0);
  AI1[26] = (AI1[25] ? AI1[24] : 0);
  AI1[27] = AI1[26] - 0;
  #pragma acc data pcopy(bodies[0:AI1[27]])
  #pragma acc kernels
  #pragma acc loop independent 
  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    b->x += dt * b->vx;
    b->y += dt * b->vy;
    b->z += dt * b->vz;
  }
}

// Generated clone.
void CPU__main__advance(int nbodies, struct planet *bodies, double dt) {
  int i, j;

  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    for (j = i + 1; j < nbodies; j++) {
      struct planet *b2 = &(bodies[j]);
      double dx = b->x - b2->x;
      double dy = b->y - b2->y;
      double dz = b->z - b2->z;
      double distance = sqrt(dx * dx + dy * dy + dz * dz);
      double mag = dt / (distance * distance * distance);
      b->vx -= dx * b2->mass * mag;
      b->vy -= dy * b2->mass * mag;
      b->vz -= dz * b2->mass * mag;
      b2->vx += dx * b->mass * mag;
      b2->vy += dy * b->mass * mag;
      b2->vz += dz * b->mass * mag;
    }
  }
  long long int AI1[28];
  AI1[0] = nbodies + -1;
  AI1[1] = 56 * AI1[0];
  AI1[2] = 16 + AI1[1];
  AI1[3] = 40 + AI1[1];
  AI1[4] = 8 + AI1[1];
  AI1[5] = 32 + AI1[1];
  AI1[6] = 24 + AI1[1];
  AI1[7] = AI1[1] > AI1[6];
  AI1[8] = (AI1[7] ? AI1[1] : AI1[6]);
  AI1[9] = AI1[1] > AI1[8];
  AI1[10] = (AI1[9] ? AI1[1] : AI1[8]);
  AI1[11] = AI1[5] > AI1[10];
  AI1[12] = (AI1[11] ? AI1[5] : AI1[10]);
  AI1[13] = AI1[4] > AI1[12];
  AI1[14] = (AI1[13] ? AI1[4] : AI1[12]);
  AI1[15] = AI1[4] > AI1[14];
  AI1[16] = (AI1[15] ? AI1[4] : AI1[14]);
  AI1[17] = AI1[3] > AI1[16];
  AI1[18] = (AI1[17] ? AI1[3] : AI1[16]);
  AI1[19] = AI1[2] > AI1[18];
  AI1[20] = (AI1[19] ? AI1[2] : AI1[18]);
  AI1[21] = AI1[2] > AI1[20];
  AI1[22] = (AI1[21] ? AI1[2] : AI1[20]);
  AI1[23] = AI1[22] + 56;
  AI1[24] = AI1[23] / 56;
  AI1[25] = (AI1[24] > 0);
  AI1[26] = (AI1[25] ? AI1[24] : 0);
  AI1[27] = AI1[26] - 0;
  #pragma acc data pcopy(bodies[0:AI1[27]])
  #pragma acc kernels
  #pragma acc loop independent 
  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    b->x += dt * b->vx;
    b->y += dt * b->vy;
    b->z += dt * b->vz;
  }
}

void advance(int nbodies, struct planet *bodies, double dt) {
  int i, j;

  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    for (j = i + 1; j < nbodies; j++) {
      struct planet *b2 = &(bodies[j]);
      double dx = b->x - b2->x;
      double dy = b->y - b2->y;
      double dz = b->z - b2->z;
      double distance = sqrt(dx * dx + dy * dy + dz * dz);
      double mag = dt / (distance * distance * distance);
      b->vx -= dx * b2->mass * mag;
      b->vy -= dy * b2->mass * mag;
      b->vz -= dz * b2->mass * mag;
      b2->vx += dx * b->mass * mag;
      b2->vy += dy * b->mass * mag;
      b2->vz += dz * b->mass * mag;
    }
  }
  long long int AI1[28];
  AI1[0] = nbodies + -1;
  AI1[1] = 56 * AI1[0];
  AI1[2] = 16 + AI1[1];
  AI1[3] = 40 + AI1[1];
  AI1[4] = 8 + AI1[1];
  AI1[5] = 32 + AI1[1];
  AI1[6] = 24 + AI1[1];
  AI1[7] = AI1[1] > AI1[6];
  AI1[8] = (AI1[7] ? AI1[1] : AI1[6]);
  AI1[9] = AI1[1] > AI1[8];
  AI1[10] = (AI1[9] ? AI1[1] : AI1[8]);
  AI1[11] = AI1[5] > AI1[10];
  AI1[12] = (AI1[11] ? AI1[5] : AI1[10]);
  AI1[13] = AI1[4] > AI1[12];
  AI1[14] = (AI1[13] ? AI1[4] : AI1[12]);
  AI1[15] = AI1[4] > AI1[14];
  AI1[16] = (AI1[15] ? AI1[4] : AI1[14]);
  AI1[17] = AI1[3] > AI1[16];
  AI1[18] = (AI1[17] ? AI1[3] : AI1[16]);
  AI1[19] = AI1[2] > AI1[18];
  AI1[20] = (AI1[19] ? AI1[2] : AI1[18]);
  AI1[21] = AI1[2] > AI1[20];
  AI1[22] = (AI1[21] ? AI1[2] : AI1[20]);
  AI1[23] = AI1[22] + 56;
  AI1[24] = AI1[23] / 56;
  AI1[25] = (AI1[24] > 0);
  AI1[26] = (AI1[25] ? AI1[24] : 0);
  AI1[27] = AI1[26] - 0;
  #pragma acc data pcopy(bodies[0:AI1[27]])
  #pragma acc kernels
  #pragma acc loop independent 
  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    b->x += dt * b->vx;
    b->y += dt * b->vy;
    b->z += dt * b->vz;
  }
}

// Generated clone.
double GPU__main__energy(int nbodies, struct planet *bodies) {
  double e;
  int i, j;

  e = 0.0;
  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    e += 0.5 * b->mass * (b->vx * b->vx + b->vy * b->vy + b->vz * b->vz);
    for (j = i + 1; j < nbodies; j++) {
      struct planet *b2 = &(bodies[j]);
      double dx = b->x - b2->x;
      double dy = b->y - b2->y;
      double dz = b->z - b2->z;
      double distance = sqrt(dx * dx + dy * dy + dz * dz);
      e -= (b->mass * b2->mass) / distance;
    }
  }
  return e;
}

// Generated clone.
double CPU__main__energy(int nbodies, struct planet *bodies) {
  double e;
  int i, j;

  e = 0.0;
  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    e += 0.5 * b->mass * (b->vx * b->vx + b->vy * b->vy + b->vz * b->vz);
    for (j = i + 1; j < nbodies; j++) {
      struct planet *b2 = &(bodies[j]);
      double dx = b->x - b2->x;
      double dy = b->y - b2->y;
      double dz = b->z - b2->z;
      double distance = sqrt(dx * dx + dy * dy + dz * dz);
      e -= (b->mass * b2->mass) / distance;
    }
  }
  return e;
}

double energy(int nbodies, struct planet *bodies) {
  double e;
  int i, j;

  e = 0.0;
  for (i = 0; i < nbodies; i++) {
    struct planet *b = &(bodies[i]);
    e += 0.5 * b->mass * (b->vx * b->vx + b->vy * b->vy + b->vz * b->vz);
    for (j = i + 1; j < nbodies; j++) {
      struct planet *b2 = &(bodies[j]);
      double dx = b->x - b2->x;
      double dy = b->y - b2->y;
      double dz = b->z - b2->z;
      double distance = sqrt(dx * dx + dy * dy + dz * dz);
      e -= (b->mass * b2->mass) / distance;
    }
  }
  return e;
}

// Generated clone.
void GPU__main__offset_momentum(int nbodies, struct planet *bodies) {
  double px = 0.0, py = 0.0, pz = 0.0;
  int i;
  for (i = 0; i < nbodies; i++) {
    px += bodies[i].vx * bodies[i].mass;
    py += bodies[i].vy * bodies[i].mass;
    pz += bodies[i].vz * bodies[i].mass;
  }
  bodies[0].vx = -px / solar_mass;
  bodies[0].vy = -py / solar_mass;
  bodies[0].vz = -pz / solar_mass;
}

// Generated clone.
void CPU__main__offset_momentum(int nbodies, struct planet *bodies) {
  double px = 0.0, py = 0.0, pz = 0.0;
  int i;
  for (i = 0; i < nbodies; i++) {
    px += bodies[i].vx * bodies[i].mass;
    py += bodies[i].vy * bodies[i].mass;
    pz += bodies[i].vz * bodies[i].mass;
  }
  bodies[0].vx = -px / solar_mass;
  bodies[0].vy = -py / solar_mass;
  bodies[0].vz = -pz / solar_mass;
}

void offset_momentum(int nbodies, struct planet *bodies) {
  double px = 0.0, py = 0.0, pz = 0.0;
  int i;
  for (i = 0; i < nbodies; i++) {
    px += bodies[i].vx * bodies[i].mass;
    py += bodies[i].vy * bodies[i].mass;
    pz += bodies[i].vz * bodies[i].mass;
  }
  bodies[0].vx = -px / solar_mass;
  bodies[0].vy = -py / solar_mass;
  bodies[0].vz = -pz / solar_mass;
}

#define NBODIES 5
struct planet bodies[NBODIES] = {
    {/* sun */
     0, 0, 0, 0, 0, 0, solar_mass},
    {/* jupiter */
     4.84143144246472090e+00, -1.16032004402742839e+00,
     -1.03622044471123109e-01, 1.66007664274403694e-03 * days_per_year,
     7.69901118419740425e-03 * days_per_year,
     -6.90460016972063023e-05 * days_per_year,
     9.54791938424326609e-04 * solar_mass},
    {/* saturn */
     8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
     -2.76742510726862411e-03 * days_per_year,
     4.99852801234917238e-03 * days_per_year,
     2.30417297573763929e-05 * days_per_year,
     2.85885980666130812e-04 * solar_mass},
    {/* uranus */
     1.28943695621391310e+01, -1.51111514016986312e+01,
     -2.23307578892655734e-01, 2.96460137564761618e-03 * days_per_year,
     2.37847173959480950e-03 * days_per_year,
     -2.96589568540237556e-05 * days_per_year,
     4.36624404335156298e-05 * solar_mass},
    {/* neptune */
     1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
     2.68067772490389322e-03 * days_per_year,
     1.62824170038242295e-03 * days_per_year,
     -9.51592254519715870e-05 * days_per_year,
     5.15138902046611451e-05 * solar_mass}};

// Generated clone.
int GPU__main(int argc, char **argv) {
  int n = 5000000;
  int i;

  offset_momentum(NBODIES, bodies);
  printf("%.9f\n", energy(NBODIES, bodies));
  for (i = 1; i <= n; i++)
    advance(NBODIES, bodies, 0.01);
  printf("%.9f\n", energy(NBODIES, bodies));
  return 0;
}

int main(int argc, char **argv) {
  int n = 5000000;
  int i;

  offset_momentum(NBODIES, bodies);
  printf("%.9f\n", energy(NBODIES, bodies));
  for (i = 1; i <= n; i++)
    advance(NBODIES, bodies, 0.01);
  printf("%.9f\n", energy(NBODIES, bodies));
  return 0;
}
