import numpy as np
import matplotlib.colors as mcol
import matplotlib.cm as cm
import pygame
import solver

class Charge:
    def __init__(self, xPos, yPos, ch_val, rad = 10, dx=0, dy=0):
        self.x = xPos
        self.y = yPos
        self.c = ch_val
        self.dx = dx
        self.dy = dy
        self.radius = rad
        self.type = "ball"
        self.color = (25, 255, 25)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self, E_x, E_y, dt):

        self.dx += E_x*self.c*dt
        self.dy += E_y*self.c*dt

        self.x += self.dx
        self.y += self.dy



def evaluate_dimensions():
    # Evaluate the width and the height of the squares.
    square_width = (window_w / ni) - line_width * ((ni + 1) / ni)
    square_height = (window_h / nj) - line_width * ((nj + 1) / nj)
    return (square_width, square_height)

def convert_column_to_x(column, square_width):
    x = line_width * (column + 1) + square_width * column
    return x

def convert_row_to_y(row, square_height):
    y = line_width * (row + 1) + square_height * row
    return y

def convert_x_to_column(x, square_width):
    return round((x-line_width)/(line_width+square_width))

def convert_y_to_row(y, square_height):
    return round((y-line_width)/(line_width+square_height))

def draw_squares():
    for row in range(nj):
        for column in range(ni):

            color = color_arr[row, column, :]  # (R, G, B)
            x = convert_column_to_x(column, square_width)
            y = convert_row_to_y(row, square_height)
            geometry = (x, y, square_width, square_height)
            pygame.draw.rect(screen, color, geometry)


black = (0, 0, 0)
white = (200, 200, 200)
red = (255, 0, 0)
pale_grey = (125, 125, 125)

ni = 50
nj = 50
dx = 1
dy = 1
window_h = 600
window_w = 600
line_width = 0.5
offset = 50

local_strength = 100

square_width, square_height = evaluate_dimensions()

color_arr = np.zeros([nj, ni, 3], dtype=np.uint8)

for i in range(ni):
    for j in range(nj):
        color_arr[i, j, :] = black

color_arr_initialized = np.zeros([nj, ni, 3], dtype=np.uint8)
color_arr_initialized[:, :, :] = color_arr

ep = solver.ElectrostaticSystem(ni, nj, square_width, square_height)

square_width, square_height = evaluate_dimensions()
screen = pygame.display.set_mode((window_h, window_w))
clock = pygame.time.Clock()
screen.fill(black)
pygame.init()

pen_down = False
ball_set = False

dirichlet_val = 0
charge_val = 1000

charges = []

# Make a user-defined colormap.
cm1 = mcol.LinearSegmentedColormap.from_list("PotentialColorMap",["y","b"])

# Make a normalizer that will map the time values from
# [start_time,end_time+1] -> [0,1].


# Turn these into an object that can be used to map time values to colors and
# can be passed to plt.colorbar().



while True:
    clock.tick(60)
    screen.fill(black)
    draw_squares()

    for i in range(len(charges)):
        if charges[i].x > window_w-offset or charges[i].x < 0 or charges[i].y >= window_h-offset or charges[i].y < 0:
            charges.pop(i)
            break

    for i in range(len(charges)):

        E_x, E_y = ep.get_field_at(charges[i].x, charges[i].y)

        charges[i].update(E_x, E_y, (1/60))
        charges[i].draw(screen)

    pygame.display.flip()

    if pen_down and not ball_set:
        x, y = pygame.mouse.get_pos()
        ep.set_dirichlet(convert_x_to_column(x, square_width), convert_y_to_row(y, square_height), dirichlet_val)
        color_arr[convert_y_to_row(y, square_height), convert_x_to_column(x, square_width), :] = red

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if not ball_set:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pen_down = True

            elif event.type == pygame.MOUSEBUTTONUP:
                pen_down = False

            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_r:
                    color_arr[:, :, :] = color_arr_initialized
                    ep.reset()
                    charges.clear()

                if key == pygame.K_i:
                    dirichlet_val = float(input("Enter Value (good vals around 1): "))

                if key == pygame.K_e:
                    ball_set = True
                    # solve
                    ep.solve()

                    cnorm = mcol.Normalize(vmin=np.min(ep.V), vmax=np.max(ep.V))
                    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
                    cpick.set_array([])

                    for row in range(nj):
                        for column in range(ni):
                            color_arr[row, column, :] = 255*np.array(cpick.to_rgba(ep.V[row, column])[:3])  # (R, G, B)


        elif ball_set:
            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_e:

                    ep.reset()

                    color_arr[:, :, :] = color_arr_initialized

                    ball_set = False

                    charges.clear()
                if key == pygame.K_i:
                    charge_val = float(input("Enter Charge Value (good vals around 1000): "))

            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                charges.append(Charge(x, y, charge_val, dx = 0))






