
from rogers.cp.pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline("conf/pipeline/training_pipeline_config.yaml", "conf/azure_config_dev.json")
    pipeline.build()
    pipeline.run()
    pipeline.publish()
    pipeline.schedule(name="CP_ReactiveTraining")
    print("pipeline_published and scheduled")


if __name__ == "__main__":
    main()