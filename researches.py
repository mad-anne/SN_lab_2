from presenter import save_plot
from neural_network import AutoEncoderMLP, DropoutMLP, L2RegularizationMLP


def save_results(train_errors, test_errors, comment):
    with open(f'./researches/dropout.txt', 'a') as results:
        results.write(f'\n\n{comment}\n')
        results.write(f'\nTrain errors\n')
        results.write(str(train_errors))
        results.write(f'\nTest errors\n')
        results.write(str(test_errors))


def save_errors_plot(epochs, train_errors, test_errors, param_name):
    epochs = [epoch + 1 for epoch in range(epochs)]
    save_plot(
        x_labels=epochs,
        y_labels_1=train_errors,
        y_labels_2=test_errors,
        x_title='liczba epok',
        y_title='błąd średniokwadratowy na wyjściu sieci [%]',
        param_name=param_name
    )


def learn_auto_encoder(params, features, train_set, test_set):
    mlp = AutoEncoderMLP(
        features=features,
        hidden_neurons=params['hiddenNeurons'],
        act_func=params['actFunc'],
        epochs=params['epochs'],
        learning_rate=params['alpha'],
        deviation=params['weightsDeviation']
    )
    train_errors, test_errors = mlp.learn(train_set=train_set, test_set=test_set)
    save_results(train_errors, test_errors, comment='Basic auto-encoder MLP')
    save_errors_plot(params['epochs'], train_errors, test_errors, 'mlp')


def learn_auto_encoder_with_dropout(params, features, train_set, test_set):
    keep_probability = params['keepProb']
    mlp = DropoutMLP(
        features=features,
        hidden_neurons=params['hiddenNeurons'],
        act_func=params['actFunc'],
        epochs=params['epochs'],
        learning_rate=params['alpha'],
        deviation=params['weightsDeviation'],
        keep_prob=keep_probability
    )
    train_errors, test_errors = mlp.learn(train_set=train_set, test_set=test_set)
    comment = f'Auto-encoder MLP with dropout keep probability = {keep_probability}'
    save_results(train_errors, test_errors, comment=comment)
    save_errors_plot(params['epochs'], train_errors, test_errors, 'mlp_dropout')


def learn_auto_encoder_with_l2_regularization(params, features, train_set, test_set):
    keep_probability = params['keepProb']
    mlp = L2RegularizationMLP(
        features=features,
        hidden_neurons=params['hiddenNeurons'],
        act_func=params['actFunc'],
        epochs=params['epochs'],
        learning_rate=params['alpha'],
        deviation=params['weightsDeviation'],
        regularization_term=params['regularizationTerm']
    )
    train_errors, test_errors = mlp.learn(train_set=train_set, test_set=test_set)
    comment = f'Auto-encoder MLP with dropout keep probability = {keep_probability}'
    save_results(train_errors, test_errors, comment=comment)
    save_errors_plot(params['epochs'], train_errors, test_errors, 'mlp_l2_reg')
