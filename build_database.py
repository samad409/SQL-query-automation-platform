import sqlite3

# 1. Initialize the Database
# This creates a local file named 'my_ai_database.db'
conn = sqlite3.connect('my_ai_database.db')
cursor = conn.cursor()

# 2. Define the Schema (Create Tables)
# We define the 5 tables perfectly matching your dataset's queries
tables = {
    "orders": """
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            status TEXT,
            date TEXT,
            amount REAL
        )
    """,
    "employees": """
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            age INTEGER,
            salary REAL
        )
    """,
    "customers": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT,
            age INTEGER,
            membership TEXT
        )
    """,
    "students": """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            age INTEGER,
            marks REAL
        )
    """,
    "products": """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """
}

# Execute table creation
for table_name, create_query in tables.items():
    cursor.execute(create_query)

# 3. Inject Sample Data
# Adding a few rows to each table so your queries will return real results
sample_data = {
    "orders": [
        (1, 101, 'Shipped', '2023-10-01', 250.50),
        (2, 102, 'Pending', '2023-10-02', 120.00),
        (3, 101, 'Delivered', '2023-10-03', 45.99)
    ],
    "employees": [
        (1, 'Alice Smith', 'Engineering', 29, 85000),
        (2, 'Bob Johnson', 'Sales', 45, 62000),
        (3, 'Charlie Brown', 'HR', 35, 71000)
    ],
    "customers": [
        (101, 'Diana Prince', 'New York', 28, 'Gold'),
        (102, 'Clark Kent', 'Metropolis', 34, 'Silver'),
        (103, 'Bruce Wayne', 'Gotham', 40, 'Platinum')
    ],
    "students": [
        (1, 'Evan Wright', 'Computer Science', 20, 88.5),
        (2, 'Fiona Gallagher', 'Mathematics', 22, 92.0),
        (3, 'George Miller', 'Physics', 21, 75.5)
    ],
    "products": [
        (1, 'Laptop', 'Electronics', 1200.00, 50),
        (2, 'Desk Chair', 'Furniture', 150.00, 200),
        (3, 'Wireless Mouse', 'Electronics', 25.50, 500)
    ]
}

try:
    # Insert data into tables
    for table_name, rows in sample_data.items():
        # Using placeholder (?) for safe insertion
        placeholders = ', '.join(['?'] * len(rows[0]))
        insert_query = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"
        cursor.executemany(insert_query, rows)
        
    conn.commit()
    print("✅ Database 'my_ai_database.db' created successfully with all tables and sample data!")

except sqlite3.Error as e:
    print(f"⚠️ An error occurred: {e}")

finally:
    conn.close()