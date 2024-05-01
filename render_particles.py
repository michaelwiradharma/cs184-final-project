import os

import bpy
import numpy as np


def setup(path):
    arr = np.load(path)

    num_timesteps = arr.shape[0]

    # Create all of the scenes
    for i in range(num_timesteps):
        new_scene = bpy.data.scenes.new(f"Scene{i}")

    # Return the loaded array.
    return num_timesteps, arr


def create_meta_ball_mesh(particles, idx, material_name, material_group, destination_folder):

    metaball_objs = []
    for particle in particles:
        bpy.ops.object.metaball_add(type="BALL", location=particle)
        metaball_objs.append(bpy.context.object)

    for metaball in metaball_objs:
        metaball_data = metaball.data
        metaball_data.materials.append(material_group)
        default_elem = metaball_data.elements[0]
        default_elem.radius = 1
    
    file_path = os.path.join(destination_folder, f"frame_{idx}_{material_name}.obj")
    
    bpy.ops.wm.obj_export(
    filepath=file_path,
    check_existing=True,
    up_axis='NEGATIVE_Z',
    forward_axis='Y',
    export_material_groups=True,
    )


#    bpy.ops.object.select_all(action='SELECT')
#    bpy.context.view_layer.objects.active = None
#    for obj in bpy.context.scene.objects:
#        if obj.type == 'META':
#            obj.select_set(True)
#    bpy.ops.object.join()


def render_frame(particles, idx, water_group, ice_group, destination_folder):

    # Set the current scene
    bpy.context.window.scene = bpy.data.scenes.get(f"Scene{idx}")

    water_particles = []
    ice_particles = []

    for i in range(particles.shape[0]):
        particle = particles[i]

        if particle[3] == 0:
            # Water particle
            water_particles.append([particle[0], particle[1], particle[2]])

        else:
            # Ice particle
            ice_particles.append([particle[0], particle[1], particle[2]])

    # Create ball mesh in this scene for both ice and water
    
    # Clear existing objects
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="META")
    bpy.ops.object.delete()
    create_meta_ball_mesh(water_particles, idx, "water", water_group, destination_folder)
    
    
    # Clear existing objects
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="META")
    bpy.ops.object.delete()
    create_meta_ball_mesh(ice_particles, idx, "ice", ice_group, destination_folder)


# FIXME: add where your data is stored.
path = "/Users/sebbyzhao/School/cs184/final-proj/cs184-final-project/point_cloud_50.npy"
destination_folder = "/Users/sebbyzhao/School/cs184/final-proj/cs184-final-project" 



# Create material groups for ice + water
water_group = bpy.data.materials.new(name="Water Group")
ice_group = bpy.data.materials.new(name="Ice Group")
# FIXME: make these reflective/change properties.

num_timesteps, arr = setup(path)
#arr = arr[:, :10, :]
#particles = arr[9]
#render_frame(particles, 9, water_group, ice_group, destination_folder)

for i in range(num_timesteps):
    particles = arr[i]

    render_frame(particles, i, water_group, ice_group, destination_folder)
    


#bpy.ops.object.metaball_add(type="BALL", location=(0, 0, 1))
#bpy.context.object.data.elements[0].radius = 1

#bpy.ops.object.metaball_add(type="BALL", location=(0, 2, 1))
#bpy.context.object.data.elements[0].radius = 1

#bpy.ops.object.metaball_add(type="BALL", location=(2, 0, 1))
#bpy.context.object.data.elements[0].radius = 1

#bpy.ops.object.metaball_add(type="BALL", location=(2, 2, 1))
#bpy.context.object.data.elements[0].radius = 1