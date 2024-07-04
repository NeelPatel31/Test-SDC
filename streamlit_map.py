import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import matplotlib.animation as animation

from ai import Dqn  # Make sure to include your Dqn class from ai.py

# Initialize variables
last_x = 0
last_y = 0
n_points = 0
length = 0

brain = Dqn(5, 3, 0.9)
action2rotation = [0, 20, -20]
last_reward = 0
scores = []

# Initialize the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global longueur
    global largeur

    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initialize last distance
last_distance = 0

def vector(x, y):
    return np.array([x, y])

def rotate(vector, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.dot(np.array([[c, -s], [s, c]]), vector)

def angle_between(v1, v2):
    return np.degrees(np.arctan2(v2[1] - v1[1], v2[0] - v1[0]))

class Car:
    def __init__(self):
        self.angle = 0
        self.rotation = 0
        self.velocity = vector(0, 0)
        self.pos = vector(0, 0)
        self.sensor1 = vector(0, 0)
        self.sensor2 = vector(0, 0)
        self.sensor3 = vector(0, 0)
        self.signal1 = 0
        self.signal2 = 0
        self.signal3 = 0

    def move(self, rotation):
        self.pos = self.velocity + self.pos
        self.rotation = rotation
        self.angle += self.rotation
        self.sensor1 = rotate(vector(30, 0), self.angle) + self.pos
        self.sensor2 = rotate(vector(30, 0), self.angle + 30) + self.pos
        self.sensor3 = rotate(vector(30, 0), self.angle - 30) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1[0]) - 10:int(self.sensor1[0]) + 10, int(self.sensor1[1]) - 10:int(self.sensor1[1]) + 10])) / 400.
        self.signal2 = int(np.sum(sand[int(self.sensor2[0]) - 10:int(self.sensor2[0]) + 10, int(self.sensor2[1]) - 10:int(self.sensor2[1]) + 10])) / 400.
        self.signal3 = int(np.sum(sand[int(self.sensor3[0]) - 10:int(self.sensor3[0]) + 10, int(self.sensor3[1]) - 10:int(self.sensor3[1]) + 10])) / 400.
        if self.sensor1[0] > longueur - 10 or self.sensor1[0] < 10 or self.sensor1[1] > largeur - 10 or self.sensor1[1] < 10:
            self.signal1 = 1.
        if self.sensor2[0] > longueur - 10 or self.sensor2[0] < 10 or self.sensor2[1] > largeur - 10 or self.sensor2[1] < 10:
            self.signal2 = 1.
        if self.sensor3[0] > longueur - 10 or self.sensor3[0] < 10 or self.sensor3[1] > largeur - 10 or self.sensor3[1] < 10:
            self.signal3 = 1.

car = Car()

fig, ax = plt.subplots()

# Define Streamlit UI components
def init_animation():
    car_circle = plt.Circle((car.pos[0], car.pos[1]), 10, color='blue')
    sensor1_circle = plt.Circle((car.sensor1[0], car.sensor1[1]), 5, color='red')
    sensor2_circle = plt.Circle((car.sensor2[0], car.sensor2[1]), 5, color='green')
    sensor3_circle = plt.Circle((car.sensor3[0], car.sensor3[1]), 5, color='yellow')
    ax.add_artist(car_circle)
    ax.add_artist(sensor1_circle)
    ax.add_artist(sensor2_circle)
    ax.add_artist(sensor3_circle)
    return car_circle, sensor1_circle, sensor2_circle, sensor3_circle

def update_animation(frame):
    global brain, last_reward, scores, last_distance, goal_x, goal_y, longueur, largeur

    if first_update:
        init()

    xx = goal_x - car.pos[0]
    yy = goal_y - car.pos[1]
    orientation = angle_between(car.velocity, vector(xx, yy)) / 180.
    last_signal = [car.signal1, car.signal2, car.signal3, orientation, -orientation]
    action = brain.update(last_reward, last_signal)
    scores.append(brain.score())
    rotation = action2rotation[action]
    car.move(rotation)
    distance = np.sqrt((car.pos[0] - goal_x) ** 2 + (car.pos[1] - goal_y) ** 2)

    if sand[int(car.pos[0]), int(car.pos[1])] > 0:
        car.velocity = rotate(vector(1, 0), car.angle)
        last_reward = -1
    else:
        car.velocity = rotate(vector(6, 0), car.angle)
        last_reward = -0.2
        if distance < last_distance:
            last_reward = 0.1

    if car.pos[0] < 10:
        car.pos[0] = 10
        last_reward = -1
    if car.pos[0] > longueur - 10:
        car.pos[0] = longueur - 10
        last_reward = -1
    if car.pos[1] < 10:
        car.pos[1] = 10
        last_reward = -1
    if car.pos[1] > largeur - 10:
        car.pos[1] = largeur - 10
        last_reward = -1

    if distance < 100:
        goal_x = longueur - goal_x
        goal_y = largeur - goal_y

    last_distance = distance

    ax.clear()
    ax.imshow(sand.T, cmap='gray')
    car_circle = plt.Circle((car.pos[0], car.pos[1]), 10, color='blue')
    sensor1_circle = plt.Circle((car.sensor1[0], car.sensor1[1]), 5, color='red')
    sensor2_circle = plt.Circle((car.sensor2[0], car.sensor2[1]), 5, color='green')
    sensor3_circle = plt.Circle((car.sensor3[0], car.sensor3[1]), 5, color='yellow')
    ax.add_artist(car_circle)
    ax.add_artist(sensor1_circle)
    ax.add_artist(sensor2_circle)
    ax.add_artist(sensor3_circle)
    plt.xlim(0, sand.shape[0])
    plt.ylim(0, sand.shape[1])
    plt.gca().invert_yaxis()
    return car_circle, sensor1_circle, sensor2_circle, sensor3_circle

# Main Streamlit loop
st.title("Self-Driving Car Simulation")

# Initialize map size
longueur = st.slider("Map width", 100, 800, 400)
largeur = st.slider("Map height", 100, 800, 400)

if st.button("Initialize"):
    init()

if st.button("Run Simulation"):
    ani = animation.FuncAnimation(fig, update_animation, init_func=init_animation, frames=200, interval=50, blit=True)
    st.pyplot(fig)

if st.button("Clear Map"):
    sand = np.zeros((longueur, largeur))
    st.write("Map cleared!")

if st.button("Save Brain"):
    brain.save()
    plt.plot(scores)
    plt.xlabel("Window size")
    plt.ylabel("Reward")
    st.pyplot(plt)

if st.button("Load Brain"):
    brain.load()
    st.write("Brain loaded!")
