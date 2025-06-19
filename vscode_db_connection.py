#setx PATH "C:\oracle\instantclient_23_8;%PATH%"
#set PATH=C:\oracle\instantclient_23_8;%PATH%


import oracledb

# Connect to the Oracle database
with oracledb.connect("SYSTEM", "oracle", "192.168.0.101:1521/FREEPDB1") as connection:
    print("✅ Connected to Oracle Database!")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM dual")
    for row in cursor:
        print(row)


