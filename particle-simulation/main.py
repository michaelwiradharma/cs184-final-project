import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FFMpegWriter
from collections import defaultdict

TIMESTEPS = 100
PLOT_EVERY = 5
NUM_PARTICLES_PER_DIMENSION = 5
BOX_SIZE = 10        # Volume enclosing all the particles.
NUM_BOXES = 100      # For hash map
MASS = 10
PARTICLE_RADIUS = 0.05
# PARTICLE_RADIUS = 0.20
RELAXATION = 100     # Relaxation parameter
EPSILON = 1e-6
DAMPING_COEFF = 0.7
GRAVITY = -9.8
p0 = 1000     # Rest density
delta_t = 0.01
SPHERE_RADIUS = BOX_SIZE / 6
SPHERE_CENTER = np.array([BOX_SIZE / 2, BOX_SIZE / 2, BOX_SIZE / 4])

# Freezing hyperparameters
FREEZING_FRACTION = 0.1
H_MAX = MASS              # Maximum virtual water film. Set to mass of liquid particle.
FREEZING_THRESHOLD = 0

particles = []
solid_particles = []
liquid_particles = []
nb_map = defaultdict(list)

def main():
    # Create liquid particles.
    # for i in range(NUM_PARTICLES_PER_DIMENSION):
    #     for j in range(NUM_PARTICLES_PER_DIMENSION):
    #         for k in range(NUM_PARTICLES_PER_DIMENSION):
    #             pos = np.array([(BOX_SIZE / 4) + i * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                             (BOX_SIZE / 4) + j * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                             (3 * BOX_SIZE / 4) + k * (BOX_SIZE / (8 * NUM_PARTICLES_PER_DIMENSION))])
    #             vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), -10.0])
    #             # vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0.0])
    #             particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
    #             particles.append(particle)
    #             liquid_particles.append(particle)

    #             pos = np.array([(BOX_SIZE / 4) + i * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)) + 0.50,
    #                             (BOX_SIZE / 4) + j * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                             (3 * BOX_SIZE / 4) + k * (BOX_SIZE / (8 * NUM_PARTICLES_PER_DIMENSION))])
    #             vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), -10.0])
    #             # vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0.0])
    #             particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
    #             particles.append(particle)
    #             liquid_particles.append(particle)

    #             pos = np.array([(BOX_SIZE / 4) + i * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                             (BOX_SIZE / 4) + j * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)) + 0.50,
    #                             (3 * BOX_SIZE / 4) + k * (BOX_SIZE / (8 * NUM_PARTICLES_PER_DIMENSION))])
    #             vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), -10.0])
    #             # vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0.0])
    #             particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
    #             particles.append(particle)
    #             liquid_particles.append(particle)

    #             pos = np.array([(BOX_SIZE / 4) + i * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                             (BOX_SIZE / 4) + j * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                             (3 * BOX_SIZE / 4) + k * (BOX_SIZE / (8 * NUM_PARTICLES_PER_DIMENSION)) + 0.25])
    #             vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), -10.0])
    #             # vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0.0])
    #             particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
    #             particles.append(particle)
    #             liquid_particles.append(particle)

    num_particles = 50
    rho = BOX_SIZE / 8
    for t in range(num_particles):
      theta = (t / num_particles) * 4 * np.pi + 0.1
      z = (BOX_SIZE / 2) + (BOX_SIZE * (t / (2 * num_particles))) - 0.75
      pos = np.array([(BOX_SIZE / 2) + rho * np.cos(theta), (BOX_SIZE / 2) +  rho * np.sin(theta), z])
      vel = np.array([0, 0, -10.0])
      particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
      particles.append(particle)
      liquid_particles.append(particle)

      theta = (t / num_particles) * 4 * np.pi + 0.2
      z = (BOX_SIZE / 2) + (BOX_SIZE * (t / (2 * num_particles))) - 0.50
      pos = np.array([(BOX_SIZE / 2) + rho * np.cos(theta), (BOX_SIZE / 2) +  rho * np.sin(theta), z])
      vel = np.array([0, 0, -10.0])
      particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
      particles.append(particle)
      liquid_particles.append(particle)

      theta = (t / num_particles) * 4 * np.pi + 0.3
      z = (BOX_SIZE / 2) + (BOX_SIZE * (t / (2 * num_particles))) - 0.25
      pos = np.array([(BOX_SIZE / 2) + rho * np.cos(theta), (BOX_SIZE / 2) +  rho * np.sin(theta), z])
      vel = np.array([0, 0, -10.0])
      particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
      particles.append(particle)
      liquid_particles.append(particle)

      theta = (t / num_particles) * 4 * np.pi
      z = (BOX_SIZE / 2) + (BOX_SIZE * (t / (2 * num_particles)))
      pos = np.array([(BOX_SIZE / 2) + rho * np.cos(theta), (BOX_SIZE / 2) +  rho * np.sin(theta), z])
      vel = np.array([0, 0, -10.0])
      particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=False)
      particles.append(particle)
      liquid_particles.append(particle)
      
    # Model solid sphere.
    rho = SPHERE_RADIUS - PARTICLE_RADIUS
    vel = np.array([0.0, 0.0, 0.0])
    # Randomly generate 500 points on the solid sphere.
    for _ in range(500):
        phi = np.pi * random.uniform(0, 1)
        theta = 2 * np.pi * random.uniform(0, 1)
        pos = SPHERE_CENTER + np.array([rho * np.sin(phi) * np.cos(theta),
                                        rho * np.sin(phi) * np.sin(theta),
                                        rho * np.cos(phi)])
        particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=True)
        particles.append(particle)
        solid_particles.append(particle)

    # Generate cube
    # Generate flat surface
    # for i in range(20):
    #   for j in range(20):
    #     pos = np.array([i * (BOX_SIZE / 20), j * (BOX_SIZE / 20), 0])
    #     particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=True)
    #     particles.append(particle)
    #     solid_particles.append(particle)
    # for _ in range(400):
    #     pos = np.array([(BOX_SIZE / 4) + i * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                     (BOX_SIZE / 4) + j * (BOX_SIZE / (2 * NUM_PARTICLES_PER_DIMENSION)),
    #                     (3 * BOX_SIZE / 4) + k * (BOX_SIZE / (8 * NUM_PARTICLES_PER_DIMENSION)) + 0.25])
    #     particle = Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=True)
    #     particles.append(particle)
    #     solid_particles.append(particle)

    # for phi in np.linspace(0, np.pi, 20):
    #   for theta in np.linspace(0, 2 * np.pi, 20):
    #     pos = np.array([(BOX_SIZE / 2) + rho * np.sin(phi) * np.cos(theta),
    #                     (BOX_SIZE / 2) + rho * np.sin(phi) * np.sin(theta),
    #                     (BOX_SIZE / 2) + rho * np.cos(phi)])
    #     particles.append(Particle(pos, vel, radius=PARTICLE_RADIUS, is_solid=True))

    # Visualization tools are based on FLIPing Fluids (Sp23 Project).
    metadata = dict(title="Position Based Fluids", artist="matlib", comment='')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    filename = f"simulations/simulation_flat_{TIMESTEPS}_{NUM_PARTICLES_PER_DIMENSION}.mp4"
    plt.style.use("dark_background")
    fig = plt.figure()

    SCATTER_DOT_SIZE = 50

    point_cloud = np.zeros((TIMESTEPS // PLOT_EVERY, len(particles), 4))

    with writer.saving(fig, filename, dpi=160):
        for t in tqdm(range(TIMESTEPS)):
            simulate()

            if t % PLOT_EVERY == 0:
                ax = plt.axes(projection="3d")
                # make the panes transparent
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                # make the grid lines transparent
                ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.set_axis_off()

                positions = np.array([p.pos for p in liquid_particles] + [p.pos for p in solid_particles])
                # positions_liquid = np.array([p.pos for p in liquid_particles])
                # positions_ice = np.array([p.pos for p in solid_particles if p.is_ice])

                states = np.array([[0] for _ in liquid_particles] + [[1] if p.is_ice else [2] for p in solid_particles])

                # Color of liquid particles depend on y-axis. Solid particles all gray color.
                # colors = np.array([[0, 0, min(1, p.pos[1] / BOX_SIZE + 0.25)] for p in liquid_particles] + [[0.7, 0.7, 0.7] for _ in solid_particles])
                colors = np.array([[0, min(1, p.pos[1] / BOX_SIZE + 0.25), 0] for p in liquid_particles] + [[1, 0, 0] if p.is_ice else [0, 0, 1] for p in solid_particles])

                # ax.scatter(
                #     positions[:, 0],
                #     positions[:, 1],
                #     positions[:, 2],
                #     s=SCATTER_DOT_SIZE,
                #     c=positions[:, 1],
                #     cmap="Blues_r"
                # )

                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    s=SCATTER_DOT_SIZE,
                    c=colors,
                )

                plt.xlim(0, BOX_SIZE)
                plt.ylim(0, BOX_SIZE)
                ax.set_zlim(0, BOX_SIZE)
                # plt.draw()
                # plt.pause(0.0001)
                writer.grab_frame()
                plt.clf()

                point_cloud[t // PLOT_EVERY] = np.concatenate((positions, states), axis=1)

    np.save(f'point_cloud/point_cloud_flat_{TIMESTEPS}_{NUM_PARTICLES_PER_DIMENSION}.npy', point_cloud)


class Particle:
  def __init__(self, pos, vel, radius, is_solid=False, is_ice=False):
    self.prev_pos = np.copy(pos)
    self.pos = pos
    self.vel = vel
    self.l = None
    self.wi = None
    self.delta_pi = None
    self.radius = radius
    self.is_solid = is_solid

    # Freezing
    self.is_ice = is_ice
    self.virtual_mass = 0
    self.growth_direction = np.zeros(3)

  def __repr__(self):
    return str(self.pos)


def hash_position(p, boxes=10):
  f_pos = p.pos / BOX_SIZE
  f_pos = f_pos * boxes
  hashed_pos = tuple(f_pos.astype(int))
  return hashed_pos


# lines 5-7 of pbf paper
def build_nb_map(ps, nb_map, boxes=10):
  nb_map.clear()
  for p in ps:
    hashed_pos = hash_position(p)
    nb_map[hashed_pos].append(p)


# lines 1-4 for pbf paper
def apply_forces(liquid_particles, gravity=-9.8):
  acceleration = np.array([0, 0, gravity])
  for p in liquid_particles:
    p.vel += delta_t * acceleration
    p.prev_pos = np.copy(p.pos)
    p.pos += delta_t * p.vel


# poly6 kernal for density estimation
def W_poly(r, h=1):
  r_dist = np.linalg.norm(r)
  if 0 <= r_dist <= h:
    left = 315 / (64 * np.pi * h ** 9)
    right = (h ** 2 - r_dist ** 2) ** 3
    return left * right
  else:
    return 0


# spiky kernel for gradients
def grad_W_spiky(r, h=1):
  r_dist = np.linalg.norm(r)
  if 0 <= r_dist <= h:
    left = -45 / (np.pi * h ** 6)
    right = (h - r_dist) ** 2
    return left * right * r / (r_dist + EPSILON)
  else:
    return np.array([0.0, 0.0, 0.0])


def handle_out_of_bounds():
  # Based on FLIPing Fluids (Sp23 Project).
  tempx = np.array([p.pos[0] for p in liquid_particles])
  tempy = np.array([p.pos[1] for p in liquid_particles])
  tempz = np.array([p.pos[2] for p in liquid_particles])

  out_of_left = tempx < EPSILON
  out_of_right = tempx > BOX_SIZE - EPSILON
  out_of_bottom = tempy < EPSILON
  out_of_top = tempy > BOX_SIZE - EPSILON
  out_of_nz = tempz < EPSILON
  out_of_pz = tempz > BOX_SIZE - EPSILON
  particlesnp = np.array(liquid_particles)

  for p in particlesnp[out_of_left]:
    p.vel[0] *= -DAMPING_COEFF
    p.pos[0] = EPSILON

  for p in particlesnp[out_of_right]:
    p.vel[0] *= -DAMPING_COEFF
    p.pos[0] = BOX_SIZE - EPSILON

  for p in particlesnp[out_of_bottom]:
    p.vel[1] *= -DAMPING_COEFF
    p.pos[1] = EPSILON

  for p in particlesnp[out_of_top]:
    p.vel[1] *= -DAMPING_COEFF
    p.pos[1] = BOX_SIZE - EPSILON

  for p in particlesnp[out_of_nz]:
    p.vel[2] *= -DAMPING_COEFF
    p.pos[2] = EPSILON

  for p in particlesnp[out_of_pz]:
    p.vel[2] *= -DAMPING_COEFF
    p.pos[2] = BOX_SIZE - EPSILON


def ray_sphere_intersection(o, d, C, R):
  # o: origin, d: direction, C: center, R: radius
  a = np.dot(d, d)
  b = 2 * np.dot(o - C, d)
  c = np.dot(o - C, o - C) - (R ** 2)
  if (b ** 2 - 4 * a * c >= 0):
    t1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a + EPSILON)
    t2 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a + EPSILON)
    if (t1 >= 0 and t2 >= 0):
      return min(t1, t2)
    if (t1 < 0) and (t2 >= 0):
      return t2
    if (t1 >= 0) and (t2 < 0):
      return t1
  return None


# def solid_liquid_collision(s, l):
#   # s: solid particle, l: liquid particle
#   o = l.prev_pos
#   d = l.pos - l.prev_pos
#   d = d / (np.linalg.norm(d) + EPSILON)
#   C = s.pos
#   R = s.radius
#   t = ray_sphere_intersection(o, d, C, R)
#   if t:
#     l.pos = o + (t - s.radius) * d
#     l.vel *= -DAMPING_COEFF
#     # TODO: Modify l's velocity.
#   return

def solid_liquid_collision(s, l):
  # s: solid particle, l: liquid particle
  o = l.prev_pos
  d = l.pos - l.prev_pos
  d = d / (np.linalg.norm(d) + EPSILON)
  C = s.pos
  R = s.radius
  t = ray_sphere_intersection(o, d, C, R)
  if t:
    l.pos = o + (t - s.radius) * d
    l.vel *= -DAMPING_COEFF
    # TODO: Modify l's velocity.
  return

def liquid_liquid_collision(p_i, p_j):
  dir = p_i.pos - p_j.pos
  length = np.linalg.norm(dir)
  if (p_i != p_j) and (length < 2 * p_i.radius):
    dir /= (length + EPSILON)
    p_i.pos += dir * (2 * p_i.radius - length) / 2
    p_j.pos -= dir * (2 * p_i.radius - length) / 2
    # TODO: Modify particles' velocities.

def handle_particle_collisions():
  for hash in nb_map:
    for p_i in nb_map[hash]:
      for p_j in nb_map[hash]:
        if (not p_i.is_solid and not p_j.is_solid):
          liquid_liquid_collision(p_i, p_j)
        elif (p_i.is_solid and not p_j.is_solid):
          solid_liquid_collision(p_i, p_j)
        elif (not p_i.is_solid and p_j.is_solid):
          solid_liquid_collision(p_j, p_i)
  return


# def sphere_liquid_collision(c, r, p):
#   # c: sphere center, r: sphere radius, p: liquid particle
#   o = p.prev_pos
#   d = p.pos - p.prev_pos
#   d = d / (np.linalg.norm(d) + EPSILON)
#   t = ray_sphere_intersection(o, d, c, r)
#   if t:
#     p.pos = o + (t * d)  # Where particle's trajectory first intersects sphere.
#     normal = p.pos - c
#     normal /= np.linalg.norm(normal)
#     incident = p.vel
#     outward = incident - (2 * np.dot(incident, normal) * normal)
#     p.vel = DAMPING_COEFF * outward

#   # Enforce that particles are not inside the solid sphere.
#   offset = p.pos - c
#   distance = np.linalg.norm(offset)
#   if distance < r:
#     offset /= distance
#     p.pos = c + (offset * (r + PARTICLE_RADIUS))
#   return


# def liquid_liquid_collision(p_i, p_j):
#   dir = p_i.pos - p_j.pos
#   length = np.linalg.norm(dir)
#   if (p_i != p_j) and (length < 2 * p_i.radius):
#     dir /= (length + EPSILON)
#     p_i.pos += dir * (2 * p_i.radius - length) / 2
#     p_j.pos -= dir * (2 * p_i.radius - length) / 2
#     # TODO: Modify particles' velocities.


# def handle_particle_collisions():
#   for hash in nb_map:
#     for p_i in nb_map[hash]:
#       if p_i.is_solid:
#         continue
#       for p_j in nb_map[hash]:
#         if p_j.is_solid:
#           continue
#         liquid_liquid_collision(p_i, p_j)
#         # elif (p_i.is_solid and not p_j.is_solid):
#         #   solid_liquid_collision(p_i, p_j)
#         # elif (not p_i.is_solid and p_j.is_solid):
#         #   solid_liquid_collision(p_j, p_i)
#   return


# lines 8-19 of pbf paper
def solver(ps, iters=1):
  for _ in range(iters):
    # lines 9-11
    for p in ps:
      h = hash_position(p)
      nbs = [other for other in nb_map[h] if p != other]
      pi = 0
      l_d = 0
      grad_ci = np.array([0.0, 0.0, 0.0])
      grad_ci_norms = 0
      for nb in nbs:
        pi += MASS * W_poly(p.pos - nb.pos)                         # Equation (2)
        grad_rst = grad_W_spiky(p.pos - nb.pos) / (p0 + EPSILON)    # Equation (7)
        grad_ci += grad_rst
        l_d += np.linalg.norm(grad_rst) ** 2

      ci = pi / p0 - 1
      l_d += np.linalg.norm(grad_ci) ** 2
      p.l = -1 * ci / (l_d + RELAXATION)                        # Equation (11)

    # lines 12-15
    for p in ps:
      delta_pi = np.array([0.0, 0.0, 0.0])
      h = hash_position(p)
      nbs = [other for other in nb_map[h] if p != other]
      for nb in nbs:
        s_corr = -0.005 * (W_poly(p.pos - nb.pos) / W_poly(np.array([0.25, 0.25, 0.25]))) ** 4   # Equation (12) tune this
        delta_pi += (p.l + nb.l + s_corr) * grad_W_spiky(p.pos - nb.pos)    # Equation (13)
      delta_pi = delta_pi / (p0 + EPSILON)
      p.delta_pi = delta_pi

    # lines 16-18
    for p in ps:
      if not p.is_solid:
        p.pos += p.delta_pi
        p.vel = (p.pos - p.prev_pos) / delta_t

    # for p in ps:
    #   if not p.is_solid:
    #     sphere_liquid_collision(SPHERE_CENTER, SPHERE_RADIUS, p)
    handle_particle_collisions()
    handle_out_of_bounds()
    return


def vorticity_viscosity_updates(ps):
  for p in ps:
    h = hash_position(p)
    # TODO: nbs = [other for other in nb_map[h] if p != other and not other.is_solid]?
    nbs = [other for other in nb_map[h] if p != other]
    wi = np.array([0.0, 0.0, 0.0])
    viscosity_sum = np.array([0.0, 0.0, 0.0])
    for nb in nbs:
      vij = nb.vel - p.vel
      grad_pj = grad_W_spiky(p.pos - nb.pos)
      W_pij = W_poly(p.pos - nb.pos)
      viscosity_sum += vij * W_pij
      wi += np.cross(vij, grad_pj)
    p.wi = wi
    # Viscosity term
    p.vel += 0.01 * viscosity_sum # tune this

  for p in ps:
    h = hash_position(p)
    nbs = [other for other in nb_map[h] if p != other]
    grad_mag_wi = np.array([0.0, 0.0, 0.0])
    for nb in nbs:
      # Approximated grad||W|| from https://joshua16266261.github.io/184-water-sim/final-report/index.html
      diff_w = np.linalg.norm(nb.wi) - np.linalg.norm(p.wi)
      diff_p = nb.pos - p.pos
      diff_p[diff_p == 0.0] = 1
      grad_mag_wi += diff_w / diff_p

    if np.linalg.norm(grad_mag_wi) != 0:
      N = grad_mag_wi / np.linalg.norm(grad_mag_wi)
      f_vorticity = 0.001 * np.cross(N, p.wi) # tune this
      # Vorticity term
      p.vel += delta_t * f_vorticity / MASS


def virtual_mass_update(solid_particles):
  for p in solid_particles:
    assert p.virtual_mass >= 0
    eta = 1 - (p.virtual_mass / H_MAX)
    h = hash_position(p)
    nbs = [other for other in nb_map[h] if p != other and not other.is_solid]
    dH = 0
    for nb in nbs:
      dH += W_poly(p.pos - nb.pos) # Equation 1
    dH *= FREEZING_FRACTION * MASS * eta
    p.virtual_mass = min(p.virtual_mass + dH, H_MAX)
    # print("eta", eta)
    # print("dH", dH)
    # print("virtual_mass", p.virtual_mass)
  return


def compute_growth_direction(solid_particles):
  for p in solid_particles:
    h = hash_position(p)
    nbs = [other for other in nb_map[h] if p != other and not other.is_solid]
    growth_direction = np.zeros(3)
    for nb in nbs:
      growth_direction += nb.vel * W_poly(p.pos - nb.pos) / len(nbs) # Equation 2
    p.growth_direction = growth_direction
  return


def compute_phase_transition(solid_particles, liquid_particles):
  for p in solid_particles:
    h = hash_position(p)
    nbs = [other for other in nb_map[h] if p != other and not other.is_solid]
    for nb in nbs:
      displacement = nb.pos - p.pos
      freezing_factor = np.dot(p.growth_direction, displacement) # Equation 3
      freezing_factor /= (np.linalg.norm(p.growth_direction) * np.linalg.norm(displacement)) + EPSILON

      nucleation_energy = p.virtual_mass * freezing_factor * W_poly(p.pos - nb.pos) # Equation 4

      # Note: For now, we ignore the difference between icicle and glaze.
      if nucleation_energy > FREEZING_THRESHOLD:
        # print("virtual_mass", p.virtual_mass)
        # print("freezing_factor", freezing_factor)
        # print("W_poly", W_poly(p.pos - nb.pos))
        # print("nucleation_energy", nucleation_energy)
        nb.is_solid = True
        nb.is_ice = True
        liquid_particles.remove(nb)
        solid_particles.append(nb)
  return


def simulate():
  apply_forces(liquid_particles, gravity=GRAVITY)
  build_nb_map(particles, nb_map, NUM_BOXES)
  solver(particles, iters=1)

  # TODO: vorticity and viscosity
  vorticity_viscosity_updates(particles)

  virtual_mass_update(solid_particles)
  compute_growth_direction(solid_particles)
  compute_phase_transition(solid_particles, liquid_particles)


if __name__ == "__main__":
    main()