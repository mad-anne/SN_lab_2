from activation_function import SigmoidFunction
from dataset import read_data_sets
from parameters import read_parameters
from researches import learn_auto_encoder, learn_auto_encoder_with_dropout, learn_auto_encoder_with_l2_regularization


if __name__ == "__main__":
    print('Reading parameters...')
    params = read_parameters('parameters.json')
    params['actFunc'] = SigmoidFunction()

    print('Reading data sets...')
    train_set, test_set = read_data_sets(params['trainingSetDir'], params['testingSetDir'])
    features = len(train_set[0].data[0])
    print(f'Size of input and output layer is {features}')

    # learn_auto_encoder(params, features, train_set, test_set)
    # learn_auto_encoder_with_dropout(params, features, train_set, test_set)
    learn_auto_encoder_with_l2_regularization(params, features, train_set, test_set)
