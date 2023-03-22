from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
import data_processing
from itpma3_utils import *
from itpma3_utils.models import MultiLayerPerceptron
from requirements import version_test


def main():
    learning_rate = 0.001
    x_train, x_val, y_train, y_val = data_processing.load_process()
    model = MultiLayerPerceptron(x_train.shape, num_classes=2, drop_rate=0.2, layers=(64, 32))
    model.build(input_shape=x_train.shape)
    model.load_weights(f'./checkpoint/{model.name}_weights.h5')

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=BinaryAccuracy()
    )

    # predictions on validation set
    y_hat_train = model.predict(x_train).ravel()
    y_hat = model.predict(x_val).ravel()
    # convert probability to predicted class
    y_hat_integer = np.floor(np.array(y_hat) + .5)

    plot_roc_auc_curve(y_val, y_hat, y_train, y_hat_train)

    # print scores
    TotalMeanMetricWrapper(model.name, average='binary', direct_cal=True)(y_val, y_hat)

    # confusion matrix
    plot_confusion_matrix(y_val, y_hat_integer)


if __name__ == "__main__":
    version_test()
    main()
