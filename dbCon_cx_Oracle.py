import oracledb
oracledb.init_oracle_client()
import cx_Oracle

dsn_tns = cx_Oracle.makedsn('192.168.0.101', '1521', service_name='freepdb1')

try:
    connection = cx_Oracle.connect(user='vector', password='vector123', dsn=dsn_tns)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM dual")
    for row in cursor:
        print(row)
except cx_Oracle.DatabaseError as e:
    print(f"An error occurred: {e}")
finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
