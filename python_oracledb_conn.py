
# This script connects to an Oracle database using oracledb and retrieves employee data
# It prints the first 10 employees with a salary greater than 5000, formatted in a table-like structure

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

