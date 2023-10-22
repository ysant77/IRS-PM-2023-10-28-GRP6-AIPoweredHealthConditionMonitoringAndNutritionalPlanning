# Note:
1. SQLite must be installed
2. Python must be installed (preferably using anaconda)

## Installation of backend:

1. (Optional) Create a new virtual environment using the command: ```conda create ENVNAME python=PYTHON-VERSION```, here ENVNAME is your environment name and PYTHON-VERSION is the desired python version (for ex: ```conda create myenv python=3.8```).
2. To optionally clone an existing environment use the command: ```conda create ENVNAME python=PYTHON-VERSION –clone EXISTING```.
3. Activate the virtual environment as: ```conda activate ENVNAME```
4. In the backend directory, install all the required packages using the command: ```pip install -r requirements.txt```
5. Install the spacy language model using the command: ```python -m spacy en_core_web_md```
6. Run the following commands to do the database migrations:
  ```
python manage.py makemigrations
python manage.py migrate

```
7. Create a superuser using the command: ```python manage.py createsuperuser``` and follow the prompts ahead.
8. Setup the Google Oauth locally as follows:
    1. Go to the URL: https://console.cloud.google.com/apis/ and create a new project.
    2. Under the APIs and Services tab click on Credentials:

