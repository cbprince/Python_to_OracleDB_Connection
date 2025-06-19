# Step 1: Install the oracledb #package pip install oracledb 
# Step 2: Install the oracle client and set path #setx PATH "C:\oracle\instantclient_23_8;%PATH%"
# Step 3: Connect to Oracle Database using oracledb package 

import oracledb

connection = oracledb.connect(
    user="system",
    password="oracle",
    dsn="192.168.0.101:1521/FREEPDB1"
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM V$VERSION")

for row in cursor:
    print(row)

cursor.close()
connection.close()