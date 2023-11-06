import os
import re
import shutil
import tarfile

import boto3

# Constants
BUCKET_NAME_DATA = "com.numidatech.modeldata"
FOLDER_IN_BUCKET_DATA = (
    ""  # This can be empty if you want to scan the root of the bucket
)
LOCAL_DIRECTORY_DATA = os.path.abspath(
    "../../../nloans_model/data/raw/pricing_data_2023_05_08"
)
FILE_PATTERN_DATA = r"^pricing-data-\d{4}-\d{2}-\d{2}\.tar\.gz$"

BUCKET_NAME_SCORECARD = "numida-prod-media"
FOLDER_IN_BUCKET_SCORECARD = (
    "scorecards"  # This can be empty if you want to scan the root of the bucket
)
LOCAL_DIRECTORY_SCORECARD = os.path.abspath("../../../nloans_model/notebooks")
FILE_PATTERN_SCORECARD = (
    r"^n-loan-scorecard-\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.pkl$"
)


# Setup the S3 client
s3 = boto3.client("s3", region_name="eu-central-1")


def fetch_latest_file_from_s3(
    bucket_name, folder_in_bucket, local_directory, file_pattern
):
    # List the objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_in_bucket)

    # Check if we have any contents
    if "Contents" not in objects:
        print("No files found in the specified S3 location.")
        return

    matching_files = [
        obj
        for obj in objects["Contents"]
        if re.match(file_pattern, os.path.basename(obj["Key"]))
    ]

    if not matching_files:
        print("No matching files found in the specified S3 location.")
        return

    # Sort the files by last modified and take the latest
    latest_file = sorted(matching_files, key=lambda x: x["LastModified"], reverse=True)[
        0
    ]
    print(f"Latest file in S3: {latest_file['Key']}")

    # Download the latest file
    local_temp_file = "temp_download"
    s3.download_file(bucket_name, latest_file["Key"], local_temp_file)

    # Check if it's a tar.gz file, then untar
    if latest_file["Key"].endswith(".tar.gz"):
        # Create a temporary extraction directory
        temp_extraction_dir = "temp_extraction"
        os.makedirs(temp_extraction_dir, exist_ok=True)

        # Extract files to the temporary directory
        with tarfile.open(local_temp_file, "r:gz") as tar_ref:
            tar_ref.extractall(temp_extraction_dir)

        # Move the extracted contents to the intended directory
        nested_folder = os.listdir(temp_extraction_dir)[
            0
        ]  # Assuming one directory is created
        for item in os.listdir(os.path.join(temp_extraction_dir, nested_folder)):
            shutil.move(
                os.path.join(temp_extraction_dir, nested_folder, item), local_directory
            )
    else:
        shutil.move(
            local_temp_file,
            os.path.join(local_directory, os.path.basename(latest_file["Key"])),
        )

    print(f"Files processed to {local_directory}")


if __name__ == "__main__":
    print("Fetching the data")

    fetch_latest_file_from_s3(
        BUCKET_NAME_DATA, FOLDER_IN_BUCKET_DATA, LOCAL_DIRECTORY_DATA, FILE_PATTERN_DATA
    )

    print("Fetching the pkl file")

    fetch_latest_file_from_s3(
        BUCKET_NAME_SCORECARD,
        FOLDER_IN_BUCKET_SCORECARD,
        LOCAL_DIRECTORY_SCORECARD,
        FILE_PATTERN_SCORECARD,
    )
