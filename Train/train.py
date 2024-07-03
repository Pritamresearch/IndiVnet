# Defining a loss object and an optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, 'tf_ckpts/', max_to_keep=3)



# Define the metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# Reference -> https://github.com/junhoning/machine_learning_tutorial/blob/b20b8a10438ec3e62f08f920744cc8ea854cde91/Visualization%20%26%20TensorBoard/%5BTensorBoard%5D%20Semantic%20Segmentation.ipynb

@tf.function
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y_train, predictions)

def train_and_checkpoint(model, manager, dataset, epoch):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    for (x_train, y_train) in dataset['train'].take(math.ceil(1403/32)):
        train_step(model, optimizer, x_train, y_train)
    ckpt.step.assign_add(1)
    save_path = manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

@tf.function
def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    test_loss(loss)
    test_accuracy(y_test, predictions)
    return predictions


# Summary writers for Tensorboard visualization
train_log_dir = 'logs/gradient_tape/train'
test_log_dir = 'logs/gradient_tape/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# This variable will help to save the best model if its performance increases after an epoch
highest_accuracy = 0

#set epoch number
EPOCH_NUMBER=15

# Training loop
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(EPOCH_NUMBER):

    print("Epoch ", epoch + 1)

    # Getting the current time before starting the training
    # This will help to keep track of how much time an epoch took
    start = time.time()

    train_and_checkpoint(model, manager, dataset, epoch + 1)

    # Saving the train loss and train accuracy metric
    train_losses.append(train_loss.result().numpy())
    train_accuracies.append(train_accuracy.result().numpy())

    # Validation phase
    for (x_test, y_test) in dataset['val'].take(math.ceil(204 / 32)):
        pred = test_step(model, x_test, y_test)

    # Saving the validation loss and validation accuracy metric
    test_losses.append(test_loss.result().numpy())
    test_accuracies.append(test_accuracy.result().numpy())

    # Calculating the time it took for the entire epoch to run
    print("Time taken ", time.time() - start)

    # Printing the metrics for the epoch
    template = 'Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'
    print(template.format(epoch + 1,
                          train_loss.result().numpy(),
                          train_accuracy.result().numpy() * 100,
                          test_loss.result().numpy(),
                          test_accuracy.result().numpy() * 100))

    # If accuracy has increased in this epoch, updating the highest accuracy and saving the model
    if test_accuracy.result().numpy() * 100 > highest_accuracy:
        print("Validation accuracy increased from {:.3f} to {:.3f}. Saving model weights.".format(highest_accuracy,
                                                                                                   test_accuracy.result().numpy() * 100))
        highest_accuracy = test_accuracy.result().numpy() * 100
        model.save_weights('IndiVnet_weights-epoch-{}.hdf5'.format(epoch + 1))

    print('_' * 80)

    # Reset metrics after every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

# Plotting the training and validation loss curves
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

# Plotting the training and validation accuracy curves
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.legend()
plt.show()


