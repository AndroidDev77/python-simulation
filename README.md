# Physics Simulation Project

This repository contains interactive physics simulations built with Python. The simulations visualize fundamental physics concepts through dynamic, real-time graphical demonstrations.

## Simulations

### 1. Orbital Simulation (`simulation.py`)

An interactive planetary orbit simulation that demonstrates gravitational physics.

**Features:**
- Interactive placement of planets with a rubber-band launch mechanism
- Realistic orbital mechanics using Newton's laws of gravitation
- Ability to adjust the sun's mass dynamically
- Orbital trajectory prediction
- Bidirectional orbit creation (clockwise or counterclockwise)
- Pan and zoom functionality

**Controls:**
- Click "Add Particle" button, then click and drag to place a planet
- Scroll wheel to zoom in/out
- Click and drag empty space to pan the view
- Use menu buttons to increase/decrease sun mass

### 2. Rock Fall Simulation (`rockfall.py`)

A simulation demonstrating the physics of a rock falling down a well, including sound propagation and air resistance effects.

**Features:**
- Accurate physics including gravity and air resistance
- Real-time visualization of falling rock and returning sound wave
- Velocity graph showing approach to terminal velocity
- Automatically calculates well depth based on round-trip time of sound

**Physics Concepts Demonstrated:**
- Terminal velocity due to air resistance
- Sound wave propagation
- Numerical integration of equations of motion

## Setup and Installation

1. Clone the repository to your local machine
2. Run the setup script to create a virtual environment and install dependencies:
   ```bash
   bash setup.sh
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Run a simulation:
   ```bash
   # For orbital simulation (requires pygame)
   pip install pygame  # if not already installed
   python simulation.py
   
   # For rock fall simulation
   python rockfall.py
   ```

## Requirements

- Python 3.6+
- numpy
- matplotlib
- pygame (for orbital simulation)

## How Physics is Modeled

### Gravitational Force
The orbital simulation uses Newton's law of universal gravitation:
```
F = G * (m1 * m2) / r^2
```
where G is the gravitational constant, m1 and m2 are the masses, and r is the distance between objects.

### Air Resistance
The rock fall simulation models air resistance as:
```
Fd = 0.5 * ρ * v^2 * Cd * A
```
where ρ is air density, v is velocity, Cd is the drag coefficient, and A is the cross-sectional area.

## Educational Value

These simulations can help students understand:
- Orbital mechanics and Kepler's laws
- The effects of air resistance on falling objects
- Terminal velocity
- Sound propagation
- Numerical integration techniques for physics simulations
