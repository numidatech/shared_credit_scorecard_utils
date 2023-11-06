# function that resolves a set of csv files with database writes, with an initial data load
# requirements:
# needs to know, the table we are resolving, the data we would like to resolve to
# should probably be a class, instantiate it with the date we want to resolve to
# instantiate it with the region we want to read from
# instantiate with the environment we want to read from
# then write method
# methods -> return data from specified table in merged form
#           writes data to intermediate spot in s3 to be able to read from again?
#           creates a intermediate folder for notebook to write into
#
#           how to clean up 
# 


"""
Output of s3.ls

s3.ls('s3://com.numidatech.datalake/staging_data_ug/core_business/', detail=True)
[{'Key': 'com.numidatech.datalake/staging_data_ug/core_business/20231023-080915478.csv',
  'LastModified': datetime.datetime(2023, 10, 23, 8, 9, 16, tzinfo=tzlocal()),
  'ETag': '"8a0208aa410f240cdf6621583e8fdd5c"',
  'Size': 1073,
  'StorageClass': 'STANDARD',
  'type': 'file',
  'size': 1073,
  'name': 'com.numidatech.datalake/staging_data_ug/core_business/20231023-080915478.csv'},
 {'Key': 'com.numidatech.datalake/staging_data_ug/core_business/20231023-082540116.csv',
  'LastModified': datetime.datetime(2023, 10, 23, 8, 25, 41, tzinfo=tzlocal()),
  'ETag': '"180659f280cf694e03e853673199d6ca"',
  'Size': 894,
  'StorageClass': 'STANDARD',
  'type': 'file',
  'size': 894,
  'name': 'com.numidatech.datalake/staging_data_ug/core_business/20231023-082540116.csv'},
 {'Key': 'com.numidatech.datalake/staging_data_ug/core_business/20231023-084448315.csv',
  'LastModified': datetime.datetime(2023, 10, 23, 8, 44, 49, tzinfo=tzlocal()),
  'ETag': '"fd1f476fb575484d129f81fd099f35a0"',
  'Size': 488,
  'StorageClass': 'STANDARD',
  'type': 'file',
  'size': 488,
  'name': 'com.numidatech.datalake/staging_data_ug/core_business/20231023-084448315.csv'},
 {'Key': 'com.numidatech.datalake/staging_data_ug/core_business/LOAD00000001.csv',
  'LastModified': datetime.datetime(2023, 10, 23, 8, 1, 25, tzinfo=tzlocal()),
  'ETag': '"af6d7fcb4bf5779e752e56ff940fc1aa"',
  'Size': 441438,
  'StorageClass': 'STANDARD',
  'type': 'file',
  'size': 441438,
  'name': 'com.numidatech.datalake/staging_data_ug/core_business/LOAD00000001.csv'}]
"""

class InvalidEnvironmentError(Exception):
    """Exception raised for unsupported environment value."""
    pass

class InvalidCountryCodeError(Exception):
    """Exception raised for unsupported country code."""
    pass


class DataLakeReader:
    SUPPORTED_ENVIRONMENTS = {'staging', 'prod'}
    SUPPORTED_COUNTRIES = {'UG', 'KE'}

    def __init__(self, date_of_data, environment, country):
        self.date_of_data = self._validate_date(data_freshness_date)
        self.environment = self._validate_environment(environment)
        self.country = self._validate_country(country)
        self.s3_prefix = f's3://com.numidatech.datalake/{environment}_data_{country.to_lower()}'
        self.s3 = s3fs.S3File()

    @staticmethod
    def _validate_date(date_value):
        try:
            date_obj = datetime.strptime(date_value, "%Y-%m-%d")
            return date_obj.date()
        except ValueError:
            raise ValueError(f"'{date_value}' is not a valid date format (YYYY-MM-DD)")

    @classmethod
    def _validate_environment(cls, environment):
        if environment not in cls.SUPPORTED_ENVIRONMENTS:
            raise InvalidEnvironmentError(f"'{environment}' is not a supported environment. Supported values are: {', '.join(cls.SUPPORTED_ENVIRONMENTS)}")
        return environment

    @classmethod
    def _validate_country(cls, country):
        if country not in cls.SUPPORTED_COUNTRIES:
            raise InvalidCountryCodeError(f"'{country}' is not a supported country code. Supported values are: {', '.join(cls.SUPPORTED_COUNTRIES)}")
        return country


    # Identifies the load file that we should use to start reconciliation from
    # If there has been multiple load files we should start with the file modified most
    # recently before the date of data that we are requesting
    def idenitfy_load_file(self, table_name, s3):
        bucket_files = s3.ls(f'{self.s3_prefix}/{table_name}', detail=True)
        closest_entry = None
        closest_diff = timedelta=(days=99999)

        for entry in bucket_files:
            if "LOAD" in entry["Key"] and entry['LastModified'].date() < self.date_of_data:
                diff = self.date_of_data - entry['LastModifed'].date()
                if diff < closest_diff:
                    closest_diff = diff
                    closest_entry = entry

        return closest_entry['Key'] if closest_entry else None

    def identify_update_files(self, table_name load_file_date):
        bucket_files = s3.ls(f'{self.s3_prefix}/{table_name}', detail=True)
        update_files = []


    def apply_updates_to_load(self, load_file, update_files):

    def read_in_table(self, table_name):
        # intialize s3fs
        load_file, load_file_date = self.identify_load_file(table_name, s3)
        update_files = self.identify_update_files(table_name, load_file_date)
