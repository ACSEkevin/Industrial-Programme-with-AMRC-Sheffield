from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from itpma3_utils.models import *
from itpma3_utils import *
import data_processing
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
from requirements import version_test


def train_keras():
    # variables declaration
    batches = 32
    epochs = 60
    learning_rate = 0.001
    input_shape = None
    save_weights = True
    x_train, x_val, y_train, y_val = data_processing.load_process()

    # model instantiation
    model = MultiLayerPerceptron(x_train.shape, num_classes=2, drop_rate=0.25, layers=(64, 32), )

    # model callbacks (checkpoint, saving weights, learning rate scheduler)
    lr_scheduler = LearningRateScheduler(
        LearningRateSchedulers(epochs, step=10, init_lr=learning_rate).linear_scheduler)

    # model compile (loss, optimizer, metrics)
    # FIXME: Loss function should be changed to the one covered in the lecture
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=BinaryAccuracy()
    )

    checkpoint = ModelCheckpoint(filepath=f'./checkpoint/{model.name}_weights.h5',
                                 monitor='val_binary_accuracy',
                                 save_best_only=True, save_weights_only=save_weights, mode='auto')

    # model train
    model._reset_compile_cache
    history = model.fit(x_train, y_train, batch_size=batches, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[lr_scheduler, checkpoint])

    # model evaluate
    train_loss, train_acc = model.evaluate(x_train, y_train)
    val_loss, val_acc = model.evaluate(x_val, y_val)

    print(f"model train loss: {train_loss}, train accuracy: {train_acc}")
    print(f"model test loss: {val_loss}, test accuracy: {val_acc}")

    # plot loss and accuracy curves
    plot_model_curve(history)

    return


def train_sklearn():
    # variables declaration

    # model instantiation

    # model fit

    # model predict

    # try model weight saving

    # plot curves

    return


if __name__ == '__main__':
    version_test()
    keras = True
    _ = train_keras() if keras is True else train_sklearn()




