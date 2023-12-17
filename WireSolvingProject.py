
import random
from collections import deque
from queue import PriorityQueue
import matplotlib.pyplot as plt
import math
import numpy as np
import copy
from itertools import product
from itertools import combinations_with_replacement

# Dimension and spreadability of the ship and its fire
d = 20

# Main method
def main ():

    images_size = 5000

    # Generate Images
    images = create_image(images_size)

    # Run the models
    model_two(images)

# Creates a database of images based on a given size
def create_image(diagrams_size):

    images = []

    # Creates diagrams_size number of images
    for i in range(diagrams_size):
        image = [[' ' for _ in range(d)] for _ in range(d)] # 2D array of closed cells, dimension x dimension large
        
        colors = ['R', 'B', 'Y', 'G'] # Discrete values for the colors

        row_indices = []
        col_indices = []

        third_wire = ''
        count = 0

        dangerous = False

        # Calculates chance of starting with a row (50%)
        is_row = True
        r = random.random()
        if r >= .5:
            is_row = False

        # Goes through each color individually laying down a wire each time
        while colors:
            rand_color = random.choice(colors)
            colors.remove(rand_color)

            if rand_color == 'R' and 'Y' not in colors:
                dangerous = True

            if count == 3 and dangerous:
                third_wire = rand_color

            if is_row:
                rand_index = random.randrange(d)
                while rand_index in row_indices:
                    rand_index = random.randrange(d)
                row_indices.append(rand_index)

                for i in range(d):
                    image[rand_index][i] = rand_color
                is_row = False
            else:
                rand_index = random.randrange(d)
                while rand_index in col_indices:
                    rand_index = random.randrange(d)
                col_indices.append(rand_index)

                for i in range(d):
                    image[i][rand_index] = rand_color
                is_row = True
            count += 1
        
        str_dang = ""
        if dangerous:
            str_dang = "D"
        else:
            str_dang = "ND"
        
        # Creates the final image array, opting for a 400 value matrix with the information about the
        # Image at the end (third wire to cut, plus if its dangerous or not)



        image.append((str_dang, third_wire))

        if diagrams_size == 1:
            images = image
        else:
            images.append(image)
    return images

# Creates a "size" number of dangerous images for model two to use
def create_dangerous_images(size):
    dangerous_images = []
    while(len(dangerous_images) < size):
        image = create_image(1)
        if image[-1][0] == 'D':
            dangerous_images.append(image)
    
    return dangerous_images

# Model one (Dangerous/Not Dangerous classification)
def model_one(images):
    
    # Input Space: 20x20 image, 400 variable data vector, values are discrete ("R", "G", "B", "Y", "")
    # Output Space: Two discrete values, either dangerous or not dangerous

    # Training data
    training_data = create_dataset(images, True, False)

    # Testing data
    testing_images = create_image(500)
    testing_data = create_dataset(testing_images, False, False)
    
    # Values alpha and lambda for gradient descent and overfitting
    a = 0.003
    l = 0.0001

    # Gradient descent to update weights
    w = logistic_gradient_descent(training_data, testing_data, a, l)

    # w = single_layer_neural_network(training_data, testing_data, a, l, 200, 40)

    # Test new data on the model
    validation_images = create_image(500)
    validation_data = create_dataset(validation_images, False, False)

    new_x = validation_data[0]
    new_y = validation_data[1]

    # Tests the data on the validation set, prints out the accuracy of the data
    s = 0
    for i in range(0, len(new_x)):
        model_output = logistic_classifcation(new_x[i], w)
        actual_output = new_y[i]

        print((actual_output, f_one(new_x[i], w)))

        if model_output == actual_output:
            s+=1
    print("Accuracy: ", s/len(validation_images))

    # print(validate_neural_network(validation_data, w[0], w[1]))

# Model two (Which wire to cut)
def model_two(images):


    # Input Space: 20x20 image, 400 variable data vector, values are discrete ("R", "G", "B", "Y", "")
    # Output Space: Which of the four color wires to cut

    # Model Space: Softmax

    # Training data
    training_images = create_dangerous_images(len(images))
    training_data = create_dataset(training_images, True, True)

    # Testing data
    testing_images = create_dangerous_images(len(images))
    testing_data = create_dataset(testing_images, False, True)
    
    # Values alpha and lambda for gradient descent and overfitting
    a = 0.003
    l = 0.0001

    # Train the data and update weights
    w = softmax_gradient_descent(training_data, testing_data, a, l)

    # Test new data on the model
    validation_images = create_dangerous_images(len(images))
    validation_data = create_dataset(validation_images, False, True)

    new_x = validation_data[0]
    new_y = validation_data[1]

    # Tests the model on new data, prints out the accuracy
    s = 0
    for i in range(0, len(new_x)):
        model_output = softmax_classification(new_x[i], w)
        actual_output = new_y[i]

        print((actual_output, model_output))

        if model_output == actual_output:
            s+=1
    print("Accuracy: ", s/len(validation_images))

# Creates the input and output dataset
def create_dataset(images, is_training, is_softmax):

    x = []

    if is_training:

        # Inflates the training dataset for more datapoints
        for i in range(0, len(images)):
            image = images[i]
            flipped_images = flip_image(image)

            for f in flipped_images:
                x.append(f)
    else:
        x = images

    # Introduces randomness so inflated data isn't grouped
    random.shuffle(x)

    # Create the expected output dataset
    y = create_training_model(x, is_softmax)

    # The input of those images
    x = convert_images(x)

    return (x, y)

# Gradient descent for the softmax training model
def softmax_gradient_descent(training_data, testing_data, a, lambda_val):

    # Threshold for cutting algorithm short and number of iterations
    threshhold = 0.01
    iterations = 1000

    # Training expected input and output
    training_x = training_data[0]
    training_y = training_data[1]   

    # Testing expected input and output
    testing_x = testing_data[0]
    testing_y = testing_data[1] 

    # Dictionary for the weights, initializes four weights for each color
    w_map = {"R": (np.random.rand(len(training_x[0])) - 0.5) * .025, "G": (np.random.rand(len(training_x[0])) - 0.5) * .025, "B": (np.random.rand(len(training_x[0])) - 0.5) * .025, "Y": (np.random.rand(len(training_x[0])) - 0.5) * .025}

    # List of loss values to be plotted later
    loss_vals = []

    for i in range(iterations):
        
        # Stochastic gradient descent 
        rand_index = random.randrange(0, len(training_x))

        rand_x = np.array(training_x[rand_index])
        rand_y = np.array(training_y[rand_index])

        index = 0
        gradient = {}
        w_old = copy.deepcopy(w_map)

        # Goes through each weight for every corresponding color
        for key in w_map.keys():
            
            # Updates the weights accordingly
            w_curr = w_old.get(key)
            gradient = a * (f_two(rand_x, w_old).get(key) - rand_y[index]) * rand_x + np.array(lambda_val * w_curr)
            w_map.update({key : np.subtract(w_curr, gradient)})
            index+=1

        # The current testing and training error
        curr_testing_error = cross_entropy_loss(testing_x, testing_y, w_map, lambda_val)
        curr_training_error = cross_entropy_loss(training_x, training_y, w_map, lambda_val)

        print((curr_training_error, curr_testing_error))
        
        # Overfitting Check
        if i > 10 and (curr_testing_error - prev_testing_error > threshhold):
            break

        loss_vals.append(cross_entropy_loss(training_x, training_y, w_map, lambda_val))

        # Updates previous errors for overfitting
        prev_training_error = curr_training_error
        prev_testing_error = curr_testing_error


    plot_loss(loss_vals)

    return w_map

# Logistic Gradient Descent training algorithm
def logistic_gradient_descent(training_data, testing_data, a, lambda_val):

    threshhold = 0.01
    iterations = 1000
    # Size of the batches for batch gradient descent
    batch_size = len(training_data) - 1

    # Values for x and y for training and testing
    training_x = training_data[0]
    training_y = training_data[1]  

    testing_x = testing_data[0]
    testing_y = testing_data[1]  

    # Initializes guess for the weights 
    w_curr = np.array((np.subtract(np.random.rand(len(training_x[0])), np.full(len(training_x[0]), 0.5))) * .025)

    loss_vals = []

    prev_training_error = 0.0
    prev_testing_error = 0.0

    for i in range(iterations):
        # Initialize gradients for all weights
        gradient_sum = np.zeros(len(w_curr))

        # Begins batch gradient descent
        batch_start = random.randrange(0, len(training_x) - batch_size)
        
        for j in range(batch_start, batch_start + batch_size):
            rand_index = random.randrange(batch_start, batch_start + batch_size)
            rand_x = np.array(training_x[rand_index])
            rand_y = np.array(training_y[rand_index])

            # Begin gradient descent, adding each gradient into a sum to be compounded into the weights
            gradient = np.array((a * (f_one(rand_x, w_curr) - rand_y)) * rand_x) + np.array(lambda_val * w_curr)

            gradient_sum = gradient_sum + np.array(gradient)
        # Update weights using the accumulated gradients
        w_curr = np.subtract(w_curr, gradient_sum / batch_size)

        # Current testing and training errors
        curr_testing_error = logistic_loss_function(testing_x, testing_y, w_curr, lambda_val)
        curr_training_error = logistic_loss_function(training_x, training_y, w_curr, lambda_val)

        # Overfitting Check
        print(curr_testing_error - prev_testing_error)
        if i > 10 and (curr_testing_error - prev_testing_error) >= threshhold:
            break

        print((curr_training_error, curr_testing_error))

        # For plotting the data
        loss_vals.append(logistic_loss_function(training_x, training_y, w_curr, lambda_val))

        # Updates previous errors for overfitting
        prev_training_error = curr_training_error
        prev_testing_error = curr_testing_error

    plot_loss(loss_vals)

    return w_curr

# Plots the loss of a given simulation
def plot_loss(loss):

    num_sims = [i for i in range(1, len(loss)+1)]

    plt.plot(num_sims, loss)

    plt.plot()

    # Labels for the axis
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.title('The loss of the model over its iterations')

    plt.legend()
    plt.show()

    return

# Logistic loss function
def logistic_loss_function(x, y, w, lambda_val):

    # Loss(fw(x), y) = −y ln [fw(x)] − (1 − y) ln [1 − fw(x)] .

    n = len(x)
    e = 1e-20
    s = 0
    for i in range(0, n-1):
        l = -1 * (y[i]) * np.log(f_one(x[i], w) + e) - ((1 - y[i]) * np.log(1 - f_one(x[i], w) + e))
        
        s += l
    
    # Lasso regulation added onto the end
    regularization_term = 0.5 * lambda_val * np.sum(np.square(w[1:]))  # Exclude the bias term from regularization
    s += regularization_term

    return s / n

# Cross entropy loss function
def cross_entropy_loss(x, y, w, lambda_val):
    n = len(x)
    epsilon = 1e-30

    s = 0
    for i in range(0, n):
        l = -1 * np.log(np.dot(y[i], list(f_two(x[i], w).values())) + epsilon)
        s += l
    # Lasso regulation added onto the end
    for i in range(0, 4):
        color = list(w)[i]
        regularization_term = 0.5 * lambda_val * np.sum(np.square(w.get(color)))
    s += regularization_term
    return s / n

# Perceptron loss function (Not used)
def perceptron_loss_function(x, y, w):

    # Loss(fw(x), y) = −y ln [fw(x)] − (1 − y) ln [1 − fw(x)] .

    n = len(x)

    s = 0
    for i in range(0, n-1):
        if f_one(x[i], w) != y[i]:
            s+=1

    return s / n

# Function for model one
def f_one(x, w):
    return sigmoid_fw(x, w)

# Function for model two, uses softmax encoding for probabilities
def f_two(x, w):
    weights = {'R': np.dot(x, w.get('R')), 'G': np.dot(x, w.get('G')), 'B': np.dot(x, w.get('B')), 'Y': np.dot(x, w.get('Y'))}
    exp_sum = e(weights['R']) + e(weights['G']) + e(weights['B']) + e(weights['Y'])

    # Weights for each color
    f_r = e(weights['R']) / exp_sum
    f_g = e(weights['G']) / exp_sum
    f_b = e(weights['B']) / exp_sum
    f_y = e(weights['Y']) / exp_sum

    # Final function as a dict
    f = {'R': f_r, 'G': f_g, 'B': f_b, 'Y': f_y}

    return f

# Initializes the expected output array for both model one or model two
def create_training_model(images, is_softmax):
    
    x = []
    color_mapping = {"R": [1, 0, 0, 0], "G": [0, 1, 0, 0], "B": [0, 0, 1, 0], "Y": [0, 0, 0, 1]}

    for image in images:
        data = image[-1]
        if is_softmax:
            x.append(color_mapping.get(data[1]))
        else:
            if data[0] == "D":
                x.append(1)
            else:
                x.append(0)

    return x

# Converts the list of images into readable one-hot encoded inputs with additional features
def convert_images(images):

    # One hot encoding
    color_mapping = {"R": [1, 0, 0, 0], "G": [0, 1, 0, 0], "B": [0, 0, 1, 0], "Y": [0, 0, 0, 1], " ": [0, 0, 0, 0]}

    # Generate all possible combinations of color_i + color_j
    combinations = list(combinations_with_replacement(color_mapping.values(), 2))

    # Converts tuples to lists
    combinations_lists = [list(np.array(combination[0]) + np.array(combination[1])) for combination in combinations]

    # The final feature dictionary for the sum of all pairs of colors
    sum_feature_dict = {}
    for c in combinations_lists:
        sum_feature_dict.update({tuple(c) : 0})

    for j in range(0, len(images)):
        images[j] = images[j][:-1]
        original_image = images[j]
        modified_image = []

        neighbor_features = []
        sum_features = []
        sum_feature_dict = sum_feature_dict.fromkeys(sum_feature_dict, 0)
        line_features = []
        row_dot = [[] for _ in range(d)]
        col_dot = [[] for _ in range(d)]

        num_colors = [0, 0, 0, 0]

        color_pair_count = [0] * 15

        # Create new image with one hot encoding
        for row in original_image:
            modified_row = []
            for cell in row:
                modified_row.append(color_mapping.get(cell, [0, 0, 0, 0]))  # Default to [0, 0, 0, 0, 1] if color not found
            modified_image.append(modified_row)

        # Features for whether a cells neighbors are the same color using dot products
        visited_cells = []
        for i in range(0, len(modified_image)):
            for k in range(0, len(modified_image[i])):
                arr = []
                sum_arr = []
                cell = modified_image[i][k]
                neighbors = find_neighbors(modified_image, i, k)

                num_colors = list(np.array(num_colors) + np.array(cell))

                # Count color pairs
                for n in neighbors:
                    neighbor_cell = modified_image[n[0]][n[1]]
                    neighbor_features.append(np.dot(neighbor_cell, cell))

                    # Feature for the sum of all pairs
                    if n not in visited_cells:
                        cell_sum = tuple(np.array(cell) + np.array(neighbor_cell))
                        curr_sum = sum_feature_dict.get(cell_sum, 0)
                        sum_feature_dict.update({cell_sum : curr_sum + 1})
                    
                visited_cells.append((i, k))

        neighbor_features = np.array(neighbor_features)

        # Adjust additional features
        sum_features = np.array(list(sum_feature_dict.values()))
        sum_features = np.array(normalize(sum_features))
        num_colors = np.subtract(np.array([d,d,d,d]), np.array(num_colors))

        # Adjust the modified image
        modified_image = np.array(modified_image)
        modified_image = modified_image.flatten()
        modified_image = np.array(list(modified_image) + list(neighbor_features) + list(sum_features) + list(num_colors))

        # Bias term
        images[j] = modified_image

    return images

# Flips an image on its axis, as well as on a 90 degree angle
# Return 8 images, each a flipped/mirrored version of the input image
def flip_image(image):
    im = image.copy()
    d = im[-1]
    im = im[:-1]

    flipped_images = [im, copy.deepcopy(mirror_image(im))]
    
    for _ in range(1, 4):
        im = rotate_image(im)
        flipped_images.append(copy.deepcopy(im))
        flipped_images.append(copy.deepcopy(mirror_image(im)))

    for f in flipped_images:
        f.append(d)

    return flipped_images

# Mirrors a given image
def mirror_image(seq):
    a = np.array(seq)
    mirrored_image = (np.flip(a, axis=0)).tolist()
    return mirrored_image

# Rotates an image 90 degrees
def rotate_image(i):
    image = i.copy()
    N = len(image[0])

    # Goes element by element, swapping them diagonally
    for row in range(0, N // 2):
        for col in range(row, N - row - 1):
            temp = image[row][col]
            image[row][col] = image[N - 1 - col][row]
            image[N - 1 - col][row] = image[N - 1 - row][N - 1 - col]
            image[N - 1 - row][N - 1 - col] = image[col][N - 1 - row]
            image[col][N - 1 - row] = temp
    return image

# Finds all neighbors of a given cell
def find_neighbors(image, row, col):
    cells = []
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Up, down, left, and right weights

    for dr, dc in directions:
        r, c = row + dr, col + dc # Row and column
        if 0 <= r < d and 0 <= c < d:
            cells.append((r,c))
    return cells

# Function for e^x
def e(x):
    if x > 100: return 1000000
    return math.exp(x)

# Function for sigmoid
def sigmoid_fw(x, w):
    return (1 / (1 + e(-1 * (np.dot(x, w)))))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Classifies a given x, w combo into a 0 or 1 based on sigmoid return value (>0.5, return 1)
def logistic_classifcation(x, w):
    if f_one(x, w) > 0.5: return 1
    return 0

# Classifies a given x, w combo into a one hot encoded return value for which wire to cut
def softmax_classification(x, w):
    max_color = ''
    max_val = 0.0

    index = 0
    max_index = 0
    l = [0, 0, 0, 0]

    f = f_two(x, w)

    for k in f.keys():
        if f.get(k) > max_val:
            max_color = k
            max_val = f.get(k)
            max_index = index
            index+=1
    
    l[max_index] = 1

    return l

# Prints out an image for debugging
def print_image(image):
    for r in range(0, len(image)):
        print(image[r])
            
# Normalizes a list of frequencies
def normalize(freq_list):
    total = sum(freq_list)
    normalized_list = [freq / total for freq in freq_list]
    return normalized_list

main()