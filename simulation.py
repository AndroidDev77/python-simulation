import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Orbital Simulation")

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Constants
G = 6.67430e-11  # Gravitational constant
SCALE = 2e8      # Scale for rendering (1 pixel = 200 million meters)
TIME_STEP = 86400  # Simulate 1 day per frame


class Particle:
    def __init__(self, mass, pos, vel, color=BLUE, radius=5):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.radius = radius

    def calculate_force(self, other):
        r_vec = other.pos - self.pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0:  # Avoid division by zero
            return np.array([0.0, 0.0])
        force_mag = G * self.mass * other.mass / r_mag**2
        force_vec = force_mag * r_vec / r_mag
        return force_vec

    def update_position(self, net_force):
        acceleration = net_force / self.mass
        self.vel += acceleration * TIME_STEP
        self.pos += self.vel * TIME_STEP

    def draw(self, screen, scale, offset):
        scaled_pos = (self.pos - offset) / scale
        pygame.draw.circle(screen, self.color, scaled_pos.astype(int), self.radius)


class Sun(Particle):
    def __init__(self, mass, pos):
        super().__init__(mass, pos, vel=[0, 0], color=YELLOW, radius=10)


class Simulation:
    def __init__(self):
        # Initialize the sun at the center of the screen in world coordinates
        self.scale = SCALE
        # Set the world coordinates for the center of the screen
        center_world_pos = np.array([0.0, 0.0])  # Sun at the origin of world coordinates
        # Place the sun at the center of the universe (in world coordinates)
        self.sun = Sun(1.989e30, center_world_pos)
        self.particles = []
        self.running = True
        self.clock = pygame.time.Clock()
        # Set offset to zero - this means we start viewing the origin
        self.offset = np.array([-WIDTH / 2 * self.scale, -HEIGHT / 2 * self.scale])
        self.adding_particle = False
        self.menu_height = 50
        self.selected_mass = 5.972e24  # Default mass for new particles
        self.dragging = False
        self.last_mouse_pos = None
        self.start_mouse_pos = None  # For calculating velocity when adding particles
        self.ghost_particle = None  # Temporary particle for visualization
        self.font = pygame.font.Font(None, 24)  # Font for rendering text

    def add_particle(self, pos, vel, mass=None):
        if mass is None:
            mass = self.selected_mass
        self.particles.append(Particle(mass, pos, vel))

    def calculate_rubber_band_velocity(self, start_pos, current_pos):
        """Calculate velocity based on rubber band effect from drag distance and direction."""
        # Vector from start to current mouse position (the drag vector)
        drag_vector = current_pos - self.start_mouse_pos
        
        # Use the current position to get the world position of the particle
        world_pos = current_pos * self.scale + self.offset
        
        # Vector from sun to particle position
        r_vec = world_pos - self.sun.pos
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag == 0:
            return np.array([0.0, 0.0])
        
        # Calculate the required orbital speed for a circular orbit
        orbit_speed = np.sqrt(G * self.sun.mass / r_mag)
        
        # Normalize the radial vector
        r_unit = r_vec / r_mag
        
        # Determine the orbital direction based on the drag direction
        # We'll use a cross product of the drag vector and the radial vector
        # in 2D: cross_z = ax*by - ay*bx
        # This effectively tells us if we're dragging clockwise or counterclockwise
        cross_z = drag_vector[0] * r_unit[1] - drag_vector[1] * r_unit[0]
        
        # If cross_z is positive, we're dragging counterclockwise
        # If cross_z is negative, we're dragging clockwise
        direction = 1 if cross_z >= 0 else -1
        
        # Create the tangential unit vector for the orbit
        # Perpendicular to the radial vector, direction determined by our drag
        tangential_unit = np.array([-direction * r_unit[1], direction * r_unit[0]])
        
        # Adjust the magnitude based on the drag distance
        drag_distance = np.linalg.norm(drag_vector)
        
        # Create a scaling factor that affects the eccentricity of the orbit
        # A value of 1.0 gives a circular orbit, <1 gives elliptical orbits
        # >1 could give escape velocity or highly eccentric orbits
        scaling = 0.8 + drag_distance / 100.0
        
        # The final velocity vector
        return tangential_unit * orbit_speed * scaling

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    if self.adding_particle:
                        # Add a ghost particle at the initial position
                        pos = mouse_pos * self.scale + self.offset
                        self.start_mouse_pos = mouse_pos
                        self.ghost_particle = Particle(self.selected_mass, pos, [0, 0], color=WHITE, radius=5)
                    elif mouse_pos[1] > HEIGHT - self.menu_height:  # Menu click
                        self.handle_menu_click(mouse_pos)
                    else:  # Start dragging
                        self.dragging = True
                        self.last_mouse_pos = mouse_pos
                elif event.button == 4:  # Scroll up (zoom in)
                    self.zoom(1.1, pygame.mouse.get_pos())
                elif event.button == 5:  # Scroll down (zoom out)
                    self.zoom(1 / 1.1, pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    if self.adding_particle and self.ghost_particle:
                        # Finalize the particle with orbital velocity based on rubber band
                        vel_vector = self.calculate_rubber_band_velocity(self.start_mouse_pos, mouse_pos)
                        self.ghost_particle.vel = vel_vector
                        self.particles.append(self.ghost_particle)
                        self.ghost_particle = None
                        self.adding_particle = False
                        self.start_mouse_pos = None
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:  # Drag the view
                    current_mouse_pos = np.array(pygame.mouse.get_pos())
                    delta = (self.last_mouse_pos - current_mouse_pos) * self.scale
                    self.offset += delta
                    self.last_mouse_pos = current_mouse_pos
                elif self.adding_particle and self.ghost_particle and self.start_mouse_pos is not None:
                    # Update ghost particle position and velocity based on mouse position
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    pos = mouse_pos * self.scale + self.offset
                    self.ghost_particle.pos = pos
                    vel_vector = self.calculate_rubber_band_velocity(self.start_mouse_pos, mouse_pos)
                    self.ghost_particle.vel = vel_vector

    def zoom(self, factor, mouse_pos):
        # Adjust scale
        old_scale = self.scale
        self.scale *= factor

        # Adjust offset to zoom around the mouse position
        mouse_world_pos = np.array(mouse_pos) * old_scale + self.offset
        self.offset = mouse_world_pos - (np.array(mouse_pos) * self.scale)

    def handle_menu_click(self, pos):
        menu_width = WIDTH // 3
        if pos[0] < menu_width:  # Add particle button
            self.adding_particle = True
        elif pos[0] < 2 * menu_width:  # Increase sun mass
            self.sun.mass *= 1.1
        else:  # Decrease sun mass
            self.sun.mass /= 1.1

    def update_particles(self):
        for particle in self.particles:
            net_force = particle.calculate_force(self.sun)
            for other in self.particles:
                if other is not particle:
                    net_force += particle.calculate_force(other)
            particle.update_position(net_force)

    def draw_trajectory(self, particle):
        """Draw the projected trajectory of the particle with proper orbital physics."""
        if not particle:
            return
            
        # Create temporary particles for simulation
        temp_particle = Particle(
            particle.mass,
            np.copy(particle.pos),
            np.copy(particle.vel),
            color=WHITE,
            radius=1
        )
        
        # Create a copy of the sun for simulation
        temp_sun = Particle(
            self.sun.mass,
            np.copy(self.sun.pos),
            np.array([0.0, 0.0]),
            color=YELLOW,
            radius=10
        )
        
        points = []
        # Simulate more steps with smaller time increments for accuracy
        sim_time_step = TIME_STEP
        steps = 500  # More steps for longer trajectory prediction
        
        for _ in range(steps):
            # Calculate gravitational force from the sun
            force = temp_particle.calculate_force(temp_sun)
            
            # Include forces from existing particles for more accurate trajectory
            for p in self.particles:
                if p is not particle:  # Skip if this is the same particle
                    temp_other = Particle(p.mass, np.copy(p.pos), np.array([0.0, 0.0]))
                    additional_force = temp_particle.calculate_force(temp_other)
                    force += additional_force
            
            # Update velocity using acceleration
            acceleration = force / temp_particle.mass
            temp_particle.vel += acceleration * sim_time_step
            
            # Update position
            temp_particle.pos += temp_particle.vel * sim_time_step
            
            # Add the new position to our trajectory points
            screen_pos = (temp_particle.pos - self.offset) / self.scale
            points.append(screen_pos)
        
        # Draw the trajectory line segments
        for i in range(len(points) - 1):
            # Gradually fade the color as the trajectory extends
            alpha = 255 - int(i * 255 / steps)
            color = (255, 255, alpha)
            pygame.draw.line(screen, color, points[i], points[i + 1], 1)

    def draw_menu(self):
        pygame.draw.rect(screen, GRAY, (0, HEIGHT - self.menu_height, WIDTH, self.menu_height))
        font = pygame.font.Font(None, 24)
        add_text = font.render("Add Particle", True, BLACK)
        increase_text = font.render("Increase Sun Mass", True, BLACK)
        decrease_text = font.render("Decrease Sun Mass", True, BLACK)
        screen.blit(add_text, (10, HEIGHT - self.menu_height + 10))
        screen.blit(increase_text, (WIDTH // 3 + 10, HEIGHT - self.menu_height + 10))
        screen.blit(decrease_text, (2 * WIDTH // 3 + 10, HEIGHT - self.menu_height + 10))

    def draw_mouse_position(self):
        # Get the current mouse position
        mouse_pos = pygame.mouse.get_pos()
        mouse_text = f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}"
        text_surface = self.font.render(mouse_text, True, WHITE)
        # Display the text in the top-right corner
        screen.blit(text_surface, (WIDTH - text_surface.get_width() - 10, 10))

    def draw_rubber_band(self):
        """Draw a line representing the rubber band effect."""
        if self.adding_particle and self.ghost_particle and self.start_mouse_pos is not None:
            current_mouse_pos = np.array(pygame.mouse.get_pos())
            
            # Calculate the rubber band strength for color intensity
            distance = np.linalg.norm(self.start_mouse_pos - current_mouse_pos)
            intensity = min(255, int(distance * 2))
            
            # Color based on rubber band strength (green to red)
            rubber_band_color = (intensity, 255 - intensity, 0)
            
            # Draw the rubber band line
            pygame.draw.line(screen, rubber_band_color, 
                            self.start_mouse_pos, 
                            current_mouse_pos, 2)
            
            # Draw a small circle at the start position for reference
            pygame.draw.circle(screen, WHITE, self.start_mouse_pos.astype(int), 3)

    def draw(self):
        screen.fill(BLACK)
        # Draw the sun - with the correct offset, it should appear at the center
        self.sun.draw(screen, self.scale, self.offset)
        # Draw all particles
        for particle in self.particles:
            particle.draw(screen, self.scale, self.offset)
        # Draw the ghost particle and its trajectory if it exists
        if self.ghost_particle:
            self.ghost_particle.draw(screen, self.scale, self.offset)
            self.draw_trajectory(self.ghost_particle)
            # Draw the rubber band effect
            self.draw_rubber_band()
        # Draw the menu
        self.draw_menu()
        # Draw the mouse position
        self.draw_mouse_position()
        # Draw coordinate axes for debugging
        center_screen = np.array([WIDTH // 2, HEIGHT // 2], dtype=int)
        pygame.draw.line(screen, (255, 0, 0), (center_screen[0] - 10, center_screen[1]), 
                        (center_screen[0] + 10, center_screen[1]), 1)  # X-axis
        pygame.draw.line(screen, (0, 255, 0), (center_screen[0], center_screen[1] - 10), 
                        (center_screen[0], center_screen[1] + 10), 1)  # Y-axis
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update_particles()
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
    pygame.quit()
