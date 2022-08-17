import argparse
from rogers.cp.pipeline import BatchInferencePipeline


def main():
    pipeline = BatchInferencePipeline(args.pipeline_config, args.azure_config)
    pipeline.build()
    pipeline.run()
    pipeline.publish()
    # pipeline.schedule()
    print("pipeline_built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("cp_inference_pipeline_run")
    parser.add_argument("--pipeline_config")
    parser.add_argument("--azure_config")
    args = parser.parse_args()
    main()
