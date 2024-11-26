import argparse
from google.cloud import aiplatform

def parse_args():
    parser = argparse.ArgumentParser(description="Set up a Vertex AI training pipeline")
    parser.add_argument(
        "--project_id",
        type=str,
        help="Project ID (get it from gcp, don't share it publicly)",
    )
    parser.add_argument(
        "--display_name",
        type=str,
        default="Test",
        help="Name of training instance",
    )
    parser.add_argument(
        "--docker_image",
        type=str,
        default="lr_image:latest",
        help="Docker Image",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="northamerica-northeast1",
        help="Location for instance, prefereably Montreal or Toronto",
    )
    return parser.parse_args()

def setup_vertex_ai_training(args):
    aiplatform.init(project=args.project_id, location=args.location, staging_bucket='gs://pm-test5')

    job = aiplatform.CustomContainerTrainingJob(
    display_name=args.display_name,
    container_uri='northamerica-northeast1-docker.pkg.dev/analytical-rain-397718/left-right-detection/lr_image:latest',
    staging_bucket='gs://pm-test5',
    location=args.location
)

    job.run(replica_count=1,
           machine_type='n1-standard-8',
           accelerator_type='NVIDIA_TESLA_P4',
           accelerator_count=1)

if __name__ == "__main__":
    args = parse_args()
    setup_vertex_ai_training(args)
