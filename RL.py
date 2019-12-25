import pygame
import random
import tensorflow as tf
import numpy as np

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
outputWrite = open("outputdata.txt", "a")

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
    player_x = display_width / 2
    player_y = display_height / 1.4
    agent_x = display_width / 2
    agent_y = agent_starting = player_y + 100
    reset_t = 400
    t = 0
    vel = 0
    click = 0

    X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Model architecture parameters
    n_stocks = 2
    n_neurons_1 = 4
    n_neurons_2 = 2
    n_target = 1
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_2, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))

    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    opt = tf.train.AdamOptimizer().minimize(mse)


    net = tf.Session()
    net.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = saver.save(net, "./trained-model.ckpt")

    saver.restore(net, "./trained-model.ckpt")
    print("Model restored.")
        # Check the values of the variables


    while not quits:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                quits = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quits = True

        if t > reset_t:
            t = 0
            agent_y = random.uniform(0, display_height)
            player_y = agent_y - 100
        else:
            vel = 5
            player_y -= vel
            pred = net.run(out, feed_dict={X: np.array([[abs(player_y - agent_y), vel]])})
            if pred[0][0] > 0:
                pred[0][0] = -pred[0][0]
            agent_y += pred[0][0]
            outputWrite.write(str(format(abs(pred[0][0]), '.3f')) + ", " + str(abs(vel)) + ", " + \
                            str(format(abs(player_y - agent_y), ".3f")) + "\n")

        if player_y < -(display_height / 20):
            agent_y = agent_starting
            player_y = agent_y - 100

        # Rendering
        gameDisplay.fill(white)
        player = movers(block_size, player_x, player_y, red)
        agent = movers(block_size, agent_x, agent_y, black)
        movingSprites = pygame.sprite.Group()
        movingSprites.add(player, agent)
        movingSprites.draw(gameDisplay)
        pygame.display.update()
        t += 1

        clock.tick(FPS)  # FPS tick



    outputWrite.write("\n")
    outputWrite.close()
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