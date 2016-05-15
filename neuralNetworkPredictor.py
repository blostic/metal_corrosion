from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import TanhLayer
import random


def read_data(filename):
    data_file = open(filename, 'r')
    lines = data_file.readlines()
    k_array = []
    v_array = []
    for line in lines:
        k, v = line.split()
        k_array.append(float(k))
        v_array.append(float(v))
    return k_array, v_array


def get_normalized_data(filename):
    k_array, v_array = read_data(filename)
    max_k = max(k_array)
    k = map(lambda x: x/max_k, k_array)
    return k, max_k, v_array


def get_data_in_brainpy_required_form(k_array, v_array):
    result = []
    for x in range(0, len(k_array)):
        result.append([k_array[x], v_array[x]])
    return result


def get_data_set(data):
    data_set = SupervisedDataSet(1, 1)
    for input, target in data:
        data_set.addSample(input, target)
        for i in range(0, 50):
            data_set.addSample(input + random.random() * 0.01 - 0.005, target + random.random() * 0.01 - 0.005)
    return data_set


def get_trained_network(data):
    data_set = get_data_set(data)

    network = buildNetwork(1, 3, 3, 1, bias=True, hiddenclass=TanhLayer)
    trainer = BackpropTrainer(network, learningrate=0.1, momentum=0.95, verbose=True)
    trainer.trainUntilConvergence(verbose=True, dataset=data_set, maxEpochs=30)

    return network


def test(trained_network, value, max_k):
    predicted_value = trained_network.activate([value/max_k])
    print value, '->', predicted_value
    return predicted_value


def main(filename):
    k, max_k, v = get_normalized_data(filename)
    data = get_data_in_brainpy_required_form(k, v)
    print data
    trained_network = get_trained_network(data)

    test(trained_network, 250, max_k)
    test(trained_network, 1250, max_k)
    test(trained_network, 2250, max_k)
    test(trained_network, 3250, max_k)


main('data/alloy_617-21')