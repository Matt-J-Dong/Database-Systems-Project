# Database-Systems-Project
Database Systems Final Project, for our company EduSolutions.

`/machine-learning-model` is where all of the code and diagrams for the machine learning analytics model is stored and run. Make sure all the code requirements from `requirements.txt` are installed first.

`textbooks.csv` is the source for the code local copy of the database. This is no longer necessary as the code connects directly to the Azure SQL Databases using SQLAlchemy and the mssql extension. It is kept to take note of previous work. However, it could technically act as a failsafe in case Azure stops working.
`purchases.csv` is deprecated in the same fashion. It used to the local storage location of the purchases, but now that data is stored in the purchases table in Microsoft Azure.

`function_app.py` is deprecated code for testing out Microsoft Azure Function applications. The one that is currently being used for the project is being hosted on Azure itself, as that is where it runs from. Check the report for more details.
`.vscode` is a folder that contains Azure setup for the functions, for similar reasons. These are also not currently being used.