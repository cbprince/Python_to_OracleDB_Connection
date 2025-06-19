# pip install cx_Oracle


import cx_Oracle

# Easy Connect string format: connection = cx_Oracle.connect("username", "password", "hostname:1521/service_name")
connection = cx_Oracle.connect("SYSTEM", "oracle","host=192.168.0.101, 1521/FREEPDB1" )

print("Connected to Oracle Database:", connection.version)
connection.close()

