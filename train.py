from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

class ModelTrainer:
    def __init__(self, model, train_gen, val_gen):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        
    def _get_class_weights(self):
        """Calculate weights for imbalanced classes"""
        y = np.concatenate([y for x, y in self.train_gen], axis=0)
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y.flatten())
        return dict(zip(classes, weights))
    
    def _temporal_consistency_loss(self, y_true, y_pred):
        """Custom loss encouraging temporal consistency"""
        # Penalizes sudden changes between consecutive timesteps
        time_diff = tf.reduce_mean(tf.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + 0.1 * time_diff
    
    def train(self, epochs=50):
        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True),
            EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=[self._temporal_consistency_loss, 'categorical_crossentropy'],
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self._get_class_weights()
        )
        
        return history
