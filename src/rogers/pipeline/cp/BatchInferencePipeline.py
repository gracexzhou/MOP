
from azureml.core import Datastore
from azureml.core.databricks import PyPiLibrary
from azureml.core.runconfig import EggLibrary
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig

from azureml.pipeline.core import Pipeline, PipelineData, Schedule, ScheduleRecurrence, TimeZone
from azureml.pipeline.steps import DatabricksStep, PythonScriptStep
from azureml.pipeline.core import PipelineEndpoint

from datetime import timezone

from src.rogers.pipeline.WLPipeline import WLPipeline


class BatchInferencePipeline(WLPipeline):

    def __init__(self, conf_file, azure_conf):
        """
        Batch Inference pipeline class

        :param conf_file: str
        The path to the pipeline config file
        :param azure_conf: str
        The path to azure config
        """

        super().__init__(conf_file, azure_conf)
        self.repo_link = self.config["repo_link"].get(str)
        self.pypi_packages = self._get_pypi_packages(self.config["pypi_packages"].get(list))
        self.schedule_frequency = self.config["schedule"]["frequency"].get(str)
        self.start_time = self.config["schedule"]["start_time"].get(str)
        self.schedule_name = self.config["schedule"]["schedule_name"].get(str)
        self.schedule_interval = self.config['schedule']['schedule_interval'].get(int)
        self.aml_schedule_timezone = self.config["schedule"]['schedule_timezone'].get(str)
        self.model_name = self.config["model_name"].get(str)
        self.output_dir = self.config["output_dir"].get(str)
        self.aml_env = self._get_or_create_env(env_name=self.config['env_name'].get(str))
        self.compute_target = self._get_aml_compute(self.config['aml_compute_target'].get(str))

        self.db_compute = self._get_db_compute(self.config['db_attached_compute'].get(str))
        self.source_directory = self.config['script_dir'].get(str)
        self.etl_step_script_name = self.config['etl_step_script_name'].get(str)
        self.infer_save_step_script_name = self.config['inference_save_step_script_name'].get(str)
        self.egg_lib = EggLibrary(library=self.config['egg_lib_path'].get(str))
        self.etl_datastore = Datastore.get(self.aml_workspace, self.config['etl_datastore'].get(str))
        self.inference_datastore = Datastore.get(self.aml_workspace, self.config['inference_datastore'].get(str))


    def build(self):
        """
        Build pipeline and steps, set self.pipeline to built pipeline
        :return:
        """
        print("execute function")
        run_config = RunConfiguration()
        run_config.target = self.compute_target
        run_config.environment = self.aml_env

        # DB Step
        etl_out_dir = PipelineData("etl_output", self.etl_datastore).as_dataset()
        etl_step = DatabricksStep(name="DB_etl_step",
                                  source_directory=self.source_directory,
                                  python_script_name=self.etl_step_script_name,
                                  run_name='cp_etl_run',
                                  compute_target=self.db_compute,
                                  allow_reuse=False,
                                  outputs=[etl_out_dir],
                                  egg_libraries=[self.egg_lib],
                                  node_type="Standard_F32s_v2",
                                  num_workers=10,
                                  spark_version="7.3.x-scala2.12",
                                  spark_env_variables={
                                      "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                                      "spark.kryoserializer.buffer.max": "2000M"},
                                  pypi_libraries=self.pypi_packages,
                                  cluster_log_dbfs_path="dbfs:/FileStore/libs/cluster_jobs/")

        # Inference and Save step
        output_dir = OutputFileDatasetConfig(name="scores_output",
                                             destination=(self.inference_datastore, self.output_dir))

        inference_save_step = PythonScriptStep(name="Inference and Save Data",
                                               source_directory=self.source_directory,
                                               script_name=self.infer_save_step_script_name,
                                               arguments=['--model_name', self.model_name,
                                                          '--output_dir', output_dir.as_upload(overwrite=True)],
                                               inputs=[etl_out_dir.parse_parquet_files()],
                                               compute_target=self.compute_target,
                                               runconfig=run_config,
                                               allow_reuse=False)

        # Run the pipeline
        pipeline = Pipeline(workspace=self.aml_workspace, steps=[etl_step, inference_save_step])
        self.pipeline = pipeline

    def publish(self):
        """
        Publish pipeline to endpoint and return published endpoint
        :return:
        """
        self.published_pipeline = self.pipeline.publish("CP_Inferencing_Pipeline", "Call Inferencing Pipeline")

        if self.endpoint is None:
            self.endpoint = PipelineEndpoint.publish(workspace=self.aml_workspace, name=self.config["endpoint_name"],
                                                     description="CP Inferencing Pipeline",
                                                     pipeline=self.published_pipeline)
            print("New Pipeline Endpoint created")
        else:
            self.endpoint.add_default(self.published_pipeline)
            print("pipeline endpoint exists, default pipeline updated")

    def schedule(self):
        """
        Create daily schedule for batch inferencing
        :return:
        """
        recurrence = ScheduleRecurrence(frequency=self.schedule_frequency,
                                        interval=self.schedule_interval, start_time=self.start_time,
                                        time_zone=getattr(TimeZone, self.aml_schedule_timezone))
        Schedule.create(self.aml_workspace, name=self.schedule_name,
                        pipeline_id=self.published_pipeline.id,
                        experiment_name=self.experiment_name, recurrence=recurrence)

    def _get_pypi_packages(self, pypi_packages):
        """
        Return list of PyPiLibraries for dbstep
        :param pypi_packages:
        :return:
        """

        return [PyPiLibrary(package=x, repo=self.repo_link) for x in pypi_packages]
