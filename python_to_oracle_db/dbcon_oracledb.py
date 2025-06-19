
# Step 1: Install the oracledb #package pip install oracledb 
# Step 2: Install the oracle client and set path #setx PATH "C:\oracle\instantclient_23_8;%PATH%"
# Step 3: Connect to Oracle Database using oracledb package
# HR SCHEMA COUNTRIES, EMPLOYEES, DEPARTMENTS, JOBS, JOB_HISTORY, LOCATIONS, REGIONS
# Public to gitHub #git --version #git init #git add . #git commit -m "Initial commit" #git remote add origin
# git config --global user.name "cbprince"
# git config --global user.email "cbprince2013@gmail.com"

import oracledb

connection = oracledb.connect(
    user="hr",
    password="welcome1##2",
    dsn="192.168.0.101:1521/FREEPDB1"
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM HR.EMPLOYEES WHERE ROWNUM <= 10 AND SALARY > 5000")

# Get column names
columns = [col[0] for col in cursor.description]

# Print header
print(" | ".join(f"{col:15}" for col in columns))
print("-" * (18 * len(columns)))

# Print rows with formatting
for row in cursor:
    print(" | ".join(f"{str(val):15}" for val in row))

cursor.close()

# git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
# git push -u origin master