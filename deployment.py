import gradio as gr
from PIL import Image

class DeploymentSystem:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, image_series):
        """Make predictions on new data"""
        processed = self.preprocessor.process(image_series)
        changes, classes = self.model.predict(processed[np.newaxis, ...])
        
        # Create visualization
        viz = self._create_visualization(classes[0], changes[0])
        return viz
    
    def launch_interface(self):
        """Create Gradio interface for demo"""
        inputs = gr.inputs.ImageSeries()
        outputs = gr.outputs.Image(type="pil")
        
        interface = gr.Interface(
            fn=self.predict,
            inputs=inputs,
            outputs=outputs,
            title="Land Use Change Detection"
        )
        
        interface.launch()
      #if __name__ == "__main__":
    # 1. Initialize and get data
    initialize_earth_engine()
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
    time_series = create_time_series(region, years=5)
    
    # 2. Preprocess data
    preprocessor = SatellitePreprocessor()
    processed_data = preprocessor.process_collection(time_series)
    
    # 3. Prepare datasets (would implement proper split)
    train_gen, val_gen, test_gen = create_dataset_splits(processed_data)
    
    # 4. Build and train model
    model = ChangeDetectionModel().build_convlstm_unet()
    trainer = ModelTrainer(model, train_gen, val_gen)
    history = trainer.train()
    
    # 5. Evaluate
    evaluator = Evaluator(model, test_gen)
    evaluator.evaluate()
    
    # 6. Deploy
    deployment = DeploymentSystem(model, preprocessor)
    deployment.launch_interface()
