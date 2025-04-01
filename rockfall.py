import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = 9.8  # gravity (m/s^2)
vs = 343  # speed of sound (m/s)
total_time = 20  # total simulation time (s)
round_trip_time = 15  # time from drop to hearing the impact (s)

# Air resistance parameters
air_density = 1.225  # kg/m^3
rock_diameter = 0.20  # m (smaller rock for less air resistance)
rock_mass = 4.0  # kg (lighter rock)
rock_cross_section = math.pi * (rock_diameter/2)**2  # m^2
drag_coefficient = 0.47  # typical for a sphere

# Calculate terminal velocity
terminal_velocity = math.sqrt(2 * rock_mass * g / (air_density * drag_coefficient * rock_cross_section))

# Calculate depth based on the round trip time (without air resistance)
# If t_fall is time to fall and t_sound is time for sound to return:
# round_trip_time = t_fall + t_sound
# t_sound = depth / vs
# depth = 0.5 * g * t_fall^2
# Substituting: round_trip_time = t_fall + (0.5 * g * t_fall^2) / vs
# This is a quadratic equation to solve for t_fall

a = 0.5 * g / vs
b = 1
c = -round_trip_time
discriminant = b**2 - 4*a*c
t_fall_no_drag = (-b + math.sqrt(discriminant)) / (2*a)

# The depth the rock would reach without air resistance
ideal_depth = 0.5 * g * t_fall_no_drag**2

print(f"Calculated ideal depth: {ideal_depth:.2f} m")
print(f"Fall time without drag: {t_fall_no_drag:.2f} s")
print(f"Sound return time: {ideal_depth/vs:.2f} s")
print(f"Theoretical round trip: {t_fall_no_drag + ideal_depth/vs:.2f} s")

# Now, we need to find the adjusted depth that will result in a 15-second round trip WITH air resistance
# We'll use a binary search approach to find this depth

def simulate_round_trip_time(test_depth):
    """Simulate the fall with air resistance and return the total round trip time."""
    # Run numerical integration to find fall time
    t = 0
    v = 0
    y = 0
    
    while y < test_depth:
        drag_force = 0.5 * air_density * v**2 * drag_coefficient * rock_cross_section
        net_force = rock_mass * g - drag_force
        a = net_force / rock_mass
        v += a * dt
        y += v * dt
        t += dt
        
        if t > 100:  # safety limit
            break
    
    # Calculate sound return time
    sound_time = test_depth / vs
    
    # Total round trip time
    return t + sound_time

# Time step for numerical integration
dt = 0.01  # s

# Binary search to find the correct depth that gives a 15-second round trip with air resistance
def find_depth_for_round_trip():
    depth_min = 10  # minimum reasonable depth (m)
    depth_max = 1000  # maximum reasonable depth (m)
    target_time = round_trip_time
    tolerance = 0.01  # acceptable error in seconds
    
    while depth_max - depth_min > 0.1:  # continue until we narrow down to within 0.1m
        mid_depth = (depth_min + depth_max) / 2
        time = simulate_round_trip_time(mid_depth)
        
        if abs(time - target_time) < tolerance:
            return mid_depth
        
        if time < target_time:
            depth_min = mid_depth
        else:
            depth_max = mid_depth
    
    return (depth_min + depth_max) / 2

# Find the adjusted depth
adjusted_depth = find_depth_for_round_trip()
print(f"Adjusted depth for 15s round trip with air resistance: {adjusted_depth:.2f} m")

# Run a detailed simulation with the adjusted depth
def simulate_fall_with_drag(target_depth):
    t = 0
    v = 0  # Initial velocity
    y = 0  # Initial position
    positions = []
    velocities = []
    times = []
    
    while y < target_depth:  # Until the rock hits the bottom
        # Calculate drag force: Fd = 0.5 * ρ * v^2 * Cd * A
        drag_force = 0.5 * air_density * v**2 * drag_coefficient * rock_cross_section
        
        # Net force is weight minus drag
        net_force = rock_mass * g - drag_force
        
        # Acceleration
        a = net_force / rock_mass
        
        # Update velocity and position using Euler integration
        v += a * dt
        y += v * dt
        t += dt
        
        positions.append(y)
        velocities.append(v)
        times.append(t)
        
        # Stop if we've exceeded a reasonable time limit
        if t > 100:  # safety limit
            break
    
    return t, y, v, times, positions, velocities

# Run the simulation with our adjusted depth
fall_time_with_drag, depth_reached, final_velocity, sim_times, sim_positions, sim_velocities = simulate_fall_with_drag(adjusted_depth)

# Calculate the actual round trip time with air resistance
sound_return_time = depth_reached / vs
actual_round_trip_time = fall_time_with_drag + sound_return_time

print(f"Final simulation: Fall time with drag: {fall_time_with_drag:.2f} s")
print(f"Final simulation: Sound return time: {sound_return_time:.2f} s")
print(f"Final simulation: Actual round trip time: {actual_round_trip_time:.2f} s")

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-1, 2)
ax.set_ylim(0, depth_reached + 50)
ax.set_xlabel("Simulation")
ax.set_ylabel("Depth (meters)")
ax.set_title(f"Rock Fall and Sound Return - Round Trip: {round_trip_time} seconds")
ax.invert_yaxis()

# Plot points
rock_point, = ax.plot([], [], 'o', color='blue', label='Rock')
sound_point, = ax.plot([], [], 'o', color='orange', label='Sound')

# UI Text placeholders
time_text = ax.text(1.05, 50, '', fontsize=10, verticalalignment='top')
rock_status_text = ax.text(1.05, 150, '', fontsize=10, verticalalignment='top')
sound_status_text = ax.text(1.05, 250, '', fontsize=10, verticalalignment='top')
rock_position_text = ax.text(1.05, 350, '', fontsize=10, verticalalignment='top')
rock_velocity_text = ax.text(1.05, 400, '', fontsize=10, verticalalignment='top')
terminal_vel_text = ax.text(1.05, 450, f'Terminal Velocity: {terminal_velocity:.2f} m/s', 
                          fontsize=10, verticalalignment='top')

# Add a vertical line to represent the well/hole
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

# Create a secondary plot for velocity profile
ax_vel = fig.add_axes([0.58, 0.15, 0.3, 0.2])  # [left, bottom, width, height]
ax_vel.set_xlabel('Time (s)')
ax_vel.set_ylabel('Velocity (m/s)')
ax_vel.set_title('Rock Velocity vs Time')
ax_vel.grid(True, linestyle='--', alpha=0.7)

# Add horizontal line for terminal velocity
ax_vel.axhline(y=terminal_velocity, color='r', linestyle='--', alpha=0.7, 
               label=f'Terminal: {terminal_velocity:.2f} m/s')
ax_vel.legend(loc='upper right', fontsize=8)

# Plot the velocity profile from our simulation
vel_line, = ax_vel.plot([], [], 'b-', label='Velocity')
ax_vel.set_xlim(0, total_time)
ax_vel.set_ylim(0, terminal_velocity * 1.1)  # Scale y-axis to terminal velocity

# Real-time frame settings
fps = 30
frames = int(total_time * fps)
t_vals = np.linspace(0, total_time, frames)

# Add a vertical line in the velocity plot to show the 15-second mark
ax_vel.axvline(x=round_trip_time, color='g', linestyle='--', alpha=0.7, label='15s mark')

# Initialization
def init():
    rock_point.set_data([], [])
    sound_point.set_data([], [])
    vel_line.set_data([], [])
    time_text.set_text('')
    rock_status_text.set_text('')
    sound_status_text.set_text('')
    rock_position_text.set_text('')
    rock_velocity_text.set_text('')
    return rock_point, sound_point, vel_line, time_text, rock_status_text, sound_status_text, rock_position_text, rock_velocity_text

# Update function
def update(frame):
    t = t_vals[frame % len(t_vals)]  # loop time
    state = {}

    # Rock position and velocity calculations with air resistance
    if t <= fall_time_with_drag:
        # Interpolate position and velocity from our simulation results
        idx = min(int(t / dt), len(sim_times) - 1)
        y_rock = sim_positions[idx]
        v_rock = sim_velocities[idx]
        state['rock'] = 'Falling'
    else:
        y_rock = depth_reached
        v_rock = 0  # Rock has stopped
        state['rock'] = 'At Bottom'

    # Sound position
    if t >= fall_time_with_drag:
        sound_time = t - fall_time_with_drag
        y_sound = depth_reached - vs * sound_time
        y_sound = max(0, y_sound)
        state['sound'] = 'Returning' if y_sound > 0 else 'Reached Top'
    else:
        y_sound = None
        state['sound'] = 'Not Yet Started'

    # Update visuals - use lists for x and y coordinates
    rock_point.set_data([0], [y_rock])  # Wrap single values in lists
    
    if y_sound is not None:
        sound_point.set_data([0.5], [y_sound])  # Wrap single values in lists
    else:
        sound_point.set_data([], [])

    # Update velocity graph
    current_time_idx = frame
    times_to_plot = t_vals[:current_time_idx+1]
    velocities_to_plot = []
    
    for plot_t in times_to_plot:
        if plot_t <= fall_time_with_drag:
            idx = min(int(plot_t / dt), len(sim_times) - 1)
            velocities_to_plot.append(sim_velocities[idx])
        else:
            velocities_to_plot.append(0)
    
    vel_line.set_data(times_to_plot, velocities_to_plot)

    # Update UI text
    time_text.set_text(f"Time Elapsed: {t:.2f} s")
    rock_status_text.set_text(f"Rock: {state['rock']}")
    sound_status_text.set_text(f"Sound: {state['sound']}")
    rock_position_text.set_text(f"Rock Depth: {y_rock:.2f} m")
    rock_velocity_text.set_text(f"Rock Velocity: {v_rock:.2f} m/s")

    return (rock_point, sound_point, vel_line, time_text, rock_status_text, 
            sound_status_text, rock_position_text, rock_velocity_text)

# Run animation
ani = animation.FuncAnimation(
    fig, update, frames=frames, init_func=init,
    blit=True, interval=1000 / fps, repeat=True
)

# Add fall time and depth information
info_text = (
    f"Round Trip Time (target): {round_trip_time:.2f} s\n"
    f"Actual Round Trip Time: {actual_round_trip_time:.2f} s\n"
    f"Fall Time (with drag): {fall_time_with_drag:.2f} s\n"
    f"Sound Return Time: {sound_return_time:.2f} s\n"
    f"Well Depth: {depth_reached:.2f} m\n"
    f"Rock Mass: {rock_mass} kg, Diameter: {rock_diameter} m\n"
    f"Terminal Velocity: {terminal_velocity:.2f} m/s"
)
ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7))

# Add a legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
