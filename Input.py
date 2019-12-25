import pygame
import random

pygame.init()

# Color tuples
black = (0, 0, 0)
white = (200, 200, 200)
red = (255, 0, 0)

# Display properties
FPS = 30
display_width = 800
display_height = int(display_width * 2 / 3)
clock = pygame.time.Clock()
block_size = display_width / 32

# File
inputWrite = open("data.csv", "a")

def message_to_screen(msg, color, fontSize, x_axis, y_axis):
    '''
    pass in the message as a string, the color tuple, the integer for font size,
    and the position of the message and the function will print it to the game's screen
    '''
    font = pygame.font.SysFont(None, fontSize)
    screen_text = font.render(msg, True, color)
    gameDisplay.blit(screen_text, [x_axis, y_axis])

# Create a game display
gameDisplay = pygame.display.set_mode([display_width, display_height])
pygame.display.set_caption("Reinforcement Learning")

def program():
    '''
    main function of the program
    '''

    # Flags
    quits = False

    # State properties
    red_x = display_width / 2
    red_y = display_height / 1.4
    black_x = display_width / 2
    black_y = agent_starting = red_y + 100
    reset_frame = 400
    frame = 0
    velocity = 0
    click = 0


    while not quits:


        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                quits = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quits = True

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x,y = pygame.mouse.get_pos()
                agent_vel = 0.1 * (y - black_y)
                black_y += agent_vel
                inputWrite.write(str(format(abs(red_y - black_y), '.1f')) + ", " + str(abs(velocity)) \
                                + ", " + str(format(abs(agent_vel), '.1f')) + "\n")
                click += 1

        if frame > reset_frame:
            frame = 0
            black_y = random.uniform(0, display_height)
            red_y = black_y - 100
        else:
            velocity = 3
            red_y -= velocity

        if red_y < -(display_height / 20):
            black_y = agent_starting
            red_y = black_y - 100

        # Rendering
        gameDisplay.fill(white)
        message_to_screen("Clicks: " + str(click), black, 40, display_width / 20, display_height / 20)
        message_to_screen("Loops: " + str(frame), black, 40, display_width / 20, display_height / 20 + 100)
        player = movers(block_size, red_x, red_y, red)
        agent = movers(block_size, black_x, black_y, black)
        movingSprites = pygame.sprite.Group()
        movingSprites.add(player, agent)
        movingSprites.draw(gameDisplay)
        pygame.display.update()
        frame += 1

        clock.tick(FPS)  # FPS tick



    inputWrite.close()
    pygame.quit()
    quit()

class asset(pygame.sprite.Sprite): # Base class for all assets in the game

    def __init__(self):

        pygame.sprite.Sprite.__init__(self)


class movers(asset): # Class for player

    def __init__(self, size, x, y, color):

        # Image properties for player
        super().__init__()
        self.image = pygame.Surface([size, size])
        self.image.fill(color)
        self.rect = self.image.get_rect()

        # Player's position
        self.rect.x = x
        self.rect.y = y

program()