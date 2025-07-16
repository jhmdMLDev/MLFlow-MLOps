from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="analytical-rain-397718", location="northamerica-northeast1")

# Define the pipeline job (assuming you have compiled it)
pipeline_job = aiplatform.PipelineJob(
    template_path="detection_train_pipeline.json",  # Path to the compiled pipeline JSON
    pipeline_root="gs://pm-test5",  # Specify a GCS path for pipeline artifacts
    display_name="scheduled-detection-training-pipeline",  # Display name for the pipeline job
)

# Define the schedule for the pipeline job
pipeline_job_schedule = aiplatform.PipelineJobSchedule(
    pipeline_job=pipeline_job,
    display_name="scheduled-detection-job",  # Display name for the schedule
)

# Create the schedule
pipeline_job_schedule.create(
    cron="*/15 * * * *",  # Schedule to run every 15 minutes
    max_concurrent_run_count=1,  # Maximum concurrent runs (optional)
    max_run_count=5,  # Maximum number of times the job can run (optional)
)

print("Scheduled the Vertex AI pipeline successfully!")
