from kfp.v2.dsl import pipeline, component
from kfp.v2 import compiler



# Define a component for the custom training job
@component
def custom_training_component():
    from google.cloud import aiplatform
    # Initialize Vertex AI (this part is correct)
    aiplatform.init(
        project="analytical-rain-397718",
        location="northamerica-northeast1",
        staging_bucket="gs://pm-test5"
    )
    job = aiplatform.CustomJob(
        display_name="schedule-detection-training",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_P4",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "northamerica-northeast1-docker.pkg.dev/analytical-rain-397718/left-right-detection/lr_image:latest",
                },
            }
        ],
    )
    job.run()  # This triggers the job to run in the custom component

# Define the pipeline
@pipeline(name="daily-detection-training-pipeline")
def detection_train_pipeline():
    custom_training_component()

# Compile the pipeline to generate the JSON definition
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=detection_train_pipeline, 
        package_path="detection_train_pipeline.json"
    )
