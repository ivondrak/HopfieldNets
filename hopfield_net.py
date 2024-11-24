import numpy as np

class HopfieldNet:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            pattern = pattern.flatten()
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def step(self, input_pattern):
        pattern = np.array(input_pattern).flatten()       
        indices = np.random.permutation(self.num_neurons)
        for i in indices:
            net_input = np.dot(self.weights[i], pattern)
            #probability = 1 / (1 + np.exp(-net_input))
            #pattern[i] = 1 if np.random.rand() < probability else -1
            pattern[i] = 1 if net_input >= 0 else -1
        return pattern.reshape(input_pattern.shape)

    def run(self, input_pattern, max_cycles=10):
        pattern = np.array(input_pattern)
        for _ in range(max_cycles):
            new_pattern = self.step(pattern)
            if np.array_equal(new_pattern, pattern):
                break
        return new_pattern

    def energy(self, pattern):
        pattern = np.array(pattern).flatten()
        return -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))
    
class BoltzmannMachine (HopfieldNet):

    def step(self, input_pattern):
        pattern = np.array(input_pattern).flatten()       
        indices = np.random.permutation(self.num_neurons)
        for i in indices:
            net_input = np.dot(self.weights[i], pattern)
            probability = 1 / (1 + np.exp(-net_input))
            pattern[i] = 1 if np.random.rand() < probability else -1
        return pattern.reshape(input_pattern.shape)
    

# Example usage:
if __name__ == "__main__":
    patterns = [[1, 1, 1, -1], [-1, -1, -1, 1]]
    hopfield_net = HopfieldNet(num_neurons=4)
    hopfield_net.train(patterns)
    
    test_pattern = [[-1, -1, 1, 1]]
    result = hopfield_net.run(test_pattern)
    print("Found pattern:", result)
    print("Energy of the pattern:", hopfield_net.energy(result))