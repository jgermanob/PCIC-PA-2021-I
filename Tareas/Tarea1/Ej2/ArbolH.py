# Import a library of functions called 'pygame'
import pygame
import math

# Initialize the game engine
pygame.init()
 
# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
 
# Set the height and width of the screen
size = [1400, 1000]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("Arbol H")
 
#Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()

def drawHTree(x, y, length, depth):
    #caso base
    if(depth == 0):
        return
    #Nota: la coordenada (0,0) esta situada en la parte superior izquierda
    #Coordenada del extremo superior izquierdo
    x0 = x - float(length/2)
    y0 = y - float(length/2)
    #Coordenada del extremo  inferior derecho
    x1 = x + float(length/2)
    y1 = y + float(length/2)
    #dibuja estructura principal (H)
    pygame.draw.line(screen, GREEN, [x0, y0], [x0,y1], 1)#Recta vertical izquierda
    pygame.draw.line(screen, GREEN, [x1, y0], [x1,y1], 1)#Recta vertical derecha
    pygame.draw.line(screen, GREEN, [x0, y], [x1,y], 1)#Recta horizontal
    #reduce la longitud de la H para la siguiente llamada recursiva
    newLength = float(length / math.sqrt(5))
    drawHTree(x0, y0, newLength, depth - 1)# Dibuja una H en la parte superior izquierda
    drawHTree(x0, y1, newLength, depth - 1)# Dibuja una H en la parte inferior izquierda
    drawHTree(x1, y0, newLength, depth - 1)# Dibuja una H en la parte superior derecha
    drawHTree(x1, y1, newLength, depth - 1)# Dibuja una H en la parte inferior derecha

#creamos la H fuera del loop para evitar que se tenga que estar dibujando en cada refresh de la pantalla
screen.fill(WHITE)
drawHTree(size[0]/2, size[1]/2, float(size[1]/2), 12)
pygame.display.flip()
while not done:

    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop

 
# Be IDLE friendly
pygame.quit()