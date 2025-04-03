import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class Evaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def evaluate(self):
        """Comprehensive model evaluation"""
        x_test, y_test = self.test_data
        y_pred = self.model.predict(x_test)
        
        # Change detection metrics
        self._plot_change_maps(y_test[0], y_pred[0])
        
        # Classification report
        y_true_class = np.argmax(y_test[1], axis=-1).flatten()
        y_pred_class = np.argmax(y_pred[1], axis=-1).flatten()
        
        print(classification_report(y_true_class, y_pred_class))
        self._plot_confusion_matrix(y_true_class, y_pred_class)
        
        # Temporal consistency analysis
        self._analyze_temporal_consistency(y_pred[0])
    
    def _plot_change_maps(self, y_true, y_pred):
        """Visualize change detection results"""
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        for t in range(5):  # Show 5 timesteps
            axes[0,t].imshow(np.argmax(y_true[0,t], axis=-1))
            axes[1,t].imshow(np.argmax(y_pred[0,t], axis=-1))
            axes[2,t].imshow(np.argmax(y_true[0,t], axis=-1) != np.argmax(y_pred[0,t], axis=-1))
        
        plt.show()
    
    def _analyze_temporal_consistency(self, pred_changes):
        """Quantify how consistent predictions are over time"""
        changes_per_pixel = np.sum(np.diff(np.argmax(pred_changes, axis=-1), axis=1) != 0, axis=1)
        print(f"Average changes per pixel: {np.mean(changes_per_pixel)}")
