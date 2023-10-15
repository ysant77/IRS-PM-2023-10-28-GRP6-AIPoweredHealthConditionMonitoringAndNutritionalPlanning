# Backend

## NOTICE
**You don't have to run migration commands before starting the server.**
**as long as the .sqlite3 file is intact.**

---



Create a new virtual environment:

```
conda create --name chatbot_ui --clone base
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

 ```
 python manage.py makemigrations
 python manage.py migrate
 ```

 Note: 
 1) SQLite must be installed.
 2) Anaconda should be installed.

Finally run the application as follows:
 (Make sure you're in the directory where manage.py is present)

```
daphne ai_health_monitoring.asgi:application
```
