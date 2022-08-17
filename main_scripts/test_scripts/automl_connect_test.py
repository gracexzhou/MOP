#from com.rogers.mlops.aml.workspace import AMLObject
from azureml.core import ComputeTarget, Datastore, Dataset, Environment, Workspace
import logging


def get_environments():
    ws = Workspace.from_config('conf/project/mop/pipeline/inferencing/dev/azure_config_dev.json')
    envs = Environment.list(ws)
    for env in envs:
        if env.startswith("AzureML"):
            print("Name", env)
            # print("packages", envs[env].python.conda_dependencies.serialize_to_string())


def get_compute_targets():
    ws = Workspace.from_config('conf/project/mop/pipeline/inferencing/dev/azure_config_dev.json')
    print("Compute Targets:")
    for compute_name in ws.compute_targets:
        compute = ws.compute_targets[compute_name]
        print("\t", compute_name, " : ", compute.type)


def get_datastores():
    ws = Workspace.from_config('conf/project/mop/pipeline/inferencing/dev/azure_config_dev.json')
    print("data stores:")
    for datastore_name in ws.datastores:
        datastore = Datastore.get(ws, datastore_name)
        print("\t", datastore_name, " : ", datastore.datastore_type)


def get_datasets():
    ws = Workspace.from_config('conf/project/mop/pipeline/inferencing/dev/azure_config_dev.json')
    print("data sets:")
    for dataset_name in list(ws.datasets.keys()):
        dataset = Dataset.get_by_name(ws, dataset_name)
        print("\t", dataset.name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    get_environments()
    get_datasets()
    get_datastores()
    get_compute_targets()
