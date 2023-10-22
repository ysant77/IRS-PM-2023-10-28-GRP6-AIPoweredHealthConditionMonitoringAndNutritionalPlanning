# Backend

 Note: 
 1) SQLite must be installed.
 2) Anaconda should be installed.

Create a new virtual environment:
ENVNAME: your environment name
[] is optional
```
conda create --name {ENVNAME} [--clone base]
activate {ENVNAME}
```

Install the requirements as follows: 
```
pip install -r requirements.txt
```
Install the spacy model as: 
```
python -m spacy download en_core_web_md
```
Then do the migrations as follows: 
(Note: You may not have to do this step before starting the server. As long as the .sqlite3 file is intact.)

 ```
 python manage.py makemigrations
 python manage.py migrate
 ```

Finally run the application as follows:
 (Make sure you're in the directory where manage.py is present)

```
python manage.py runserver
```
(or run "daphne ai_health_monitoring.asgi:application")