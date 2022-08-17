from rogers.pipeline.mop.TrainingPipeline import TrainingPipeline

def main():
    pipeline = TrainingPipeline("/Users/Dipesh.Patel/MLWirelineInitiatives/conf/project/mop/pipeline/training/training_pipeline_config.yml",
                                "/Users/Dipesh.Patel/MLWirelineInitiatives/conf/project/mop/pipeline/inferencing/dev/azure_config_dev.json")
    pipeline.build()
    pipeline.run()
    # pipeline.publish()
    # pipeline.schedule(name="MOP_ReactiveTraining")
    print("pipeline_published")
    print("pipeline run successful")


if __name__ == "__main__":
    main()