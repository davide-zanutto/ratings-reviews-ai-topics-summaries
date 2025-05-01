import wandb

# Initialize a new W&B run
run = wandb.init(
    project="topic-assignment",   # replace with your project name
    job_type="upload_csv_artifact"
)

# Create a new Artifact to hold CSV files
artifact = wandb.Artifact(
    name="streamlit",    # artifact name
    type="dataset",           # artifact type (e.g., "dataset")
)

# Add local CSV files to the artifact
artifact.add_file("csv/images.csv")  # path to your first CSV
artifact.add_file("csv/GroundTruthProdArea10kV3.csv")  # path to your second CSV

# Log (upload) the artifact to W&B
run.log_artifact(artifact)

# Optionally, wait until the artifact is uploaded before ending the run
artifact.wait()

# Finish the W&B run
run.finish()
