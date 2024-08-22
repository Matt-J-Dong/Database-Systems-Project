# Database-Systems-Project
Database Systems Final Project, for our company EduSolutions.

`/machine-learning-model` is where all of the code and diagrams for the machine learning analytics model is stored and run. Make sure all the code requirements from `requirements.txt` are installed first.

`textbooks.csv` is the source for the code local copy of the database. This is no longer necessary as the code connects directly to the Azure SQL Databases using SQLAlchemy and the mssql extension. It is kept to take note of previous work. However, it could technically act as a failsafe in case Azure stops working.
`purchases.csv` is deprecated in the same fashion. It used to the local storage location of the purchases, but now that data is stored in the purchases table in Microsoft Azure.
