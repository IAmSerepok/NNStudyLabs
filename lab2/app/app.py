import pygame
import os
import sys

import numpy as np

from tensorflow.keras.models import load_model


if os.name == 'nt': 
    import msvcrt
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    sys.stdout.reconfigure(encoding='utf-8') 

pygame.init()

CELL_SIZE = 25
GRID_SIZE = 28
WIDTH, HEIGHT = CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Рисовалка')

model = load_model('model/model.keras') 

def clear_grid():
    return np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)


def draw_circle(grid, pos, radius=2):
    for y in range(max(0, pos[1] - radius), min(GRID_SIZE, pos[1] + radius + 1)):
        for x in range(max(0, pos[0] - radius), min(GRID_SIZE, pos[0] + radius+1)):
            distance = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            if distance <= radius:

                col = np.max([0, 255 - int(distance * 110), grid[y, x][0]]) 
                grid[y, x] = (col, col, col)


def draw_grid(grid):
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            pygame.draw.rect(screen, grid[y, x], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            
def preprocess_image(grid):
    image = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            intensity = grid[y, x, 0]
            image[y, x] = intensity
    image = image.reshape(1, 28, 28) / 255.0
    return image


def predict(grid):
    input_image = preprocess_image(grid)
    prediction = model.predict(input_image) 
    predicted_class = np.argmax(prediction) 
    return predicted_class


def draw():
    grid = clear_grid()
    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    drawing = True
                if event.button == 3:  
                    grid = clear_grid()

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    
                predicted_class = predict(grid)  
                pygame.display.set_caption(f'Рисовалка: {predicted_class}')

        pos = pygame.mouse.get_pos()
        grid_pos = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE

        if drawing:
            draw_circle(grid, grid_pos)

        screen.fill(WHITE)
        draw_grid(grid)
        pygame.display.flip()


if __name__ == "__main__":
    draw()
