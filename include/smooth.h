/**
 * Kalman - Kalman filter in constant acceleration model
 * PID - PID control plus deadzone and saturation
 * Usage in run.c
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define abs(a) ((a) > 0 ? (a) : -(a))

struct Kalman_state {
  int z_prev;             // previous input
  double x[3];            // state
  double P[3][3];         // estimate error
  double Q[3][3];         // process noise
  double R;               // measurement noise
};

struct Pid_state {
  double zs;		// smoothed input
  double gain_prev;	// previous gain
  double err, err_prev;	// error, previous error
  double integral;	// integral of error
  double kp, ki, kd;	// params
  double sat, dead, rat;
  double dsat, ddead, drat;
};

struct Face_region_state {
  struct Kalman_state *skx;
  struct Kalman_state *sky;
  struct Kalman_state *skw;
  struct Kalman_state *skh;
  struct Pid_state *spx;
  struct Pid_state *spy;
  struct Pid_state *spw;
  struct Pid_state *sph;
};

struct Face_region {
  int x, y;		// coordinate of top-left corner
  int w, h;		// width and height of the box
};

void kalman_init(struct Kalman_state *s, double *param) {
  // A[3][3] = {{1,1,0},{0,1,1},{0,0,1}};
  // H[3] = {1,0,0};
  // parameters
  memset(s->P, 0, sizeof(double) * 9);
  memset(s->Q, 0, sizeof(double) * 9);
  s->P[0][0] = param[0];
  s->P[1][1] = param[1];
  s->P[2][2] = param[2];
  s->Q[0][0] = param[3];
  s->Q[1][1] = param[4];
  s->Q[2][2] = param[5];
  s->R = param[6];
  // other init value
  s->x[0] = -10000;
  s->x[1] = 0;
  s->x[2] = 0;
}

int kalman_correct(struct Kalman_state *s, int z) {
  // set the initial x
  if (s->x[0] < -9999) return s->x[0] = z;
  // assume z keeps constant when no valid input
  if (z < 1) z = s->z_prev;
  s->z_prev = z;
  // x = A * x
  s->x[0] = s->x[0] + s->x[1];
  s->x[1] = s->x[1] + s->x[2];
  // P = A * P * A' + Q
  s->P[0][0] = s->P[0][0] + s->P[0][1] + s->P[1][0] + s->P[1][1] + s->Q[0][0];
  s->P[0][1] = s->P[0][1] + s->P[0][2] + s->P[1][1] + s->P[1][2] + s->Q[0][1];
  s->P[0][2] = s->P[0][2] + s->P[1][2] + s->Q[0][2];
  s->P[1][0] = s->P[1][0] + s->P[1][1] + s->P[2][0] + s->P[2][1] + s->Q[1][0];
  s->P[1][1] = s->P[1][1] + s->P[1][2] + s->P[2][1] + s->P[2][2] + s->Q[1][1];
  s->P[1][2] = s->P[1][2] + s->P[2][2] + s->Q[1][2];
  s->P[2][0] = s->P[2][0] + s->P[2][1] + s->Q[2][0];
  s->P[2][1] = s->P[2][1] + s->P[2][2] + s->Q[2][1];
  s->P[2][2] = s->P[2][2] + s->Q[2][2];
  // K = P * H' / (H * P * H' + R)
  //   = P * H' / (P[0][0] + R)
  double K[3], tmp = s->P[0][0] + s->R;
  K[0] = s->P[0][0] / tmp;
  K[1] = s->P[1][0] / tmp;
  K[2] = s->P[2][0] / tmp;
  // x = x + K * (z - H * x)
  double tmp1 = z - s->x[0];
  s->x[0] = s->x[0] + K[0] * tmp1;
  s->x[1] = s->x[1] + K[1] * tmp1;
  s->x[2] = s->x[2] + K[2] * tmp1;
  // P = P - K * H * P
  s->P[1][0] = s->P[1][0] - K[1] * s->P[0][0];
  s->P[1][1] = s->P[1][1] - K[1] * s->P[0][1];
  s->P[1][2] = s->P[1][2] - K[1] * s->P[0][2];
  s->P[2][0] = s->P[2][0] - K[2] * s->P[0][0];
  s->P[2][1] = s->P[2][1] - K[2] * s->P[0][1];
  s->P[2][2] = s->P[2][2] - K[2] * s->P[0][2];
  s->P[0][0] = s->P[0][0] - K[0] * s->P[0][0];
  s->P[0][1] = s->P[0][1] - K[0] * s->P[0][1];
  s->P[0][2] = s->P[0][2] - K[0] * s->P[0][2];

  return (int)(s->x[0]);
}

void pid_init(struct Pid_state *s, double *param) {
  s->kp = param[0];
  s->ki = param[1];
  s->kd = param[2];
  s->sat = param[3];
  s->dead = param[4];
  s->rat = param[5];
  s->dsat = param[6];
  s->ddead = param[7];
  s->drat = param[8];
  // other init value
  s->gain_prev = 0;
  s->err = 0;
  s->err_prev = 0;
  s->integral = 0;
  s->zs = -10000;
}

int pid_smoothe(struct Pid_state *s, int z) {
  double g, dg; // gain, diff of gain
  double zs; // smoothed input

  // set initial zs
  if (s->zs < -9999) return s->zs = z;

  // calc gain by PID
  g = s->kp * s->err + s->kd * (s->err - s->err_prev) + s->ki * s->integral;

  // apply deadzone and saturation on diff of gain
  dg = g - s->gain_prev;
  dg = dg + 0.5 * (1 - s->drat) * (abs(dg - s->ddead) - abs(dg + s->ddead));
  dg = max(-(s->dsat), min(s->dsat, dg));
  g = s->gain_prev + dg;
  // apply deadzone and saturation on gain
  g = g + 0.5 * (1 - s->rat) * (abs(g - s->dead) - abs(g + s->dead));
  g = max(-(s->sat), min(s->sat, g));

  // get smoothe input
  zs = s->zs + g;

  // update state
  s->err_prev = s->err;
  s->err = z - zs;
  s->integral += s->err;
  s->gain_prev = g;
  s->zs = zs;

  return (int)(zs);
}

int face_region_init(struct Face_region_state *s) {
  FILE *fp;
  char line[1024];
  double param0[7], param1[7], param2[9], param3[9];
  int i;

  s->skx = (struct Kalman_state *)malloc(sizeof(struct Kalman_state));
  s->sky = (struct Kalman_state *)malloc(sizeof(struct Kalman_state));
  s->skw = (struct Kalman_state *)malloc(sizeof(struct Kalman_state));
  s->skh = (struct Kalman_state *)malloc(sizeof(struct Kalman_state));
  s->spx = (struct Pid_state *)malloc(sizeof(struct Pid_state));
  s->spy = (struct Pid_state *)malloc(sizeof(struct Pid_state));
  s->spw = (struct Pid_state *)malloc(sizeof(struct Pid_state));
  s->sph = (struct Pid_state *)malloc(sizeof(struct Pid_state));

  // load params from smoothe.cfg
  if ((fp = fopen("./data/cfg/smoothe.cfg", "r")) == NULL) {
    printf("ERROR: smoothe.cfg not exist\n");
    exit(EXIT_FAILURE);
  }
  fgets(line, 1024, fp);
  param0[0] = atof(strtok(line, " "));
  for (i = 1; i < 7; i++)
    param0[i] = atof(strtok(NULL, " "));
  fgets(line, 1024, fp);
  param1[0] = atof(strtok(line, " "));
  for (i = 1; i < 7; i++)
    param1[i] = atof(strtok(NULL, " "));
  fgets(line, 1024, fp);
  param2[0] = atof(strtok(line, " "));
  for (i = 1; i < 9; i++)
    param2[i] = atof(strtok(NULL, " "));
  fgets(line, 1024, fp);
  param3[0] = atof(strtok(line, " "));
  for (i = 1; i < 9; i++)
    param3[i] = atof(strtok(NULL, " "));
  fclose(fp);

  kalman_init(s->skx, param0);
  kalman_init(s->sky, param0);
  kalman_init(s->skw, param1);
  kalman_init(s->skh, param1);
  pid_init(s->spx, param2);
  pid_init(s->spy, param2);
  pid_init(s->spw, param3);
  pid_init(s->sph, param3);

  return 1;
}

void face_region_clean(struct Face_region_state *s) {
  free(s->skx);
  free(s->sky);
  free(s->skw);
  free(s->skh);
  free(s->spx);
  free(s->spy);
  free(s->spw);
  free(s->sph);
  free(s);
}

struct Face_region face_region_get(int x, int y, int w, int h) {
  struct Face_region r;
  r.x = x;
  r.y = y;
  r.w = w;
  r.h = h;
  return r;
}

struct Face_region face_region_smoothe(struct Face_region_state *s,
                                       struct Face_region r) {
  struct Face_region rs;
  rs.w = pid_smoothe(s->spw, kalman_correct(s->skw, r.w));
  rs.h = pid_smoothe(s->sph, kalman_correct(s->skh, r.h));
  rs.x = pid_smoothe(s->spx, kalman_correct(s->skx, r.x + r.w / 2)) - rs.w / 2;
  rs.y = pid_smoothe(s->spy, kalman_correct(s->sky, r.y + r.h / 2)) - rs.h / 2;
  return rs;
}
