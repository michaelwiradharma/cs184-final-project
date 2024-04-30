import bpy
import os
import numpy as np

def setup(path):
    arr = np.load(path)
    
    # Test with only a small subset.
#    arr = arr[:, :, :]
    
    num_timesteps = arr.shape[0]
    
    # Create all of the scenes
    for i in range(num_timesteps):
        new_scene = bpy.data.scenes.new(f"Scene{i}")
    
    # Return the loaded array.
    return num_timesteps, arr


def create_meta_ball_mesh(particles, material_group):
    
    metaball_objs = []
    for particle in particles:
        bpy.ops.object.metaball_add(type='BALL', location=particle)
        metaball_objs.append(bpy.context.object)
        
    for metaball in metaball_objs:
            metaball_data = metaball.data
            metaball_data.materials.append(material_group)
            default_elem = metaball_data.elements[0]
            default_elem.radius = .3
              
#    bpy.ops.object.select_all(action='SELECT')
#    bpy.context.view_layer.objects.active = None
#    for obj in bpy.context.scene.objects:
#        if obj.type == 'META':
#            obj.select_set(True)
#    bpy.ops.object.join()
    
def render_frame(particles, idx, water_group, ice_group):
    
    # Set the current scene
    bpy.context.window.scene = bpy.data.scenes.get(f"Scene{idx}")
    
    water_particles = []
    ice_particles = []

    for i in range(arr.shape[0]):  
        particle = particles[i]
        
        if particle[3] == 0:
            #Water particle
            water_particles.append([particle[0], particle[1], -particle[2]])
            
        else:
            # Ice particle
            ice_particles.append([particle[0], particle[1], -particle[2]])
    
    # Create ball mesh in this scene for both ice and water
    create_meta_ball_mesh(water_particles, water_group)
    create_meta_ball_mesh(ice_particles, ice_group)
    


#FIXME: add where your data is stored.
path = "/Users/sebbyzhao/School/cs184/final-proj/cs184-final-project/point_cloud_50.npy" 

    
# Clear existing objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='META')
bpy.ops.object.delete()

# Create material groups for ice + water
water_group = bpy.data.materials.new(name="Water Group")
ice_group = bpy.data.materials.new(name="Ice Group")
#FIXME: make these reflective/change properties.

num_timesteps, arr = setup(path)
for i in range(num_timesteps):
    particles = arr[i]
    
    render_frame(particles, i, water_group, ice_group)






        
        