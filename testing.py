from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///data/airline.sqlite")

print(db.get_usable_table_names())

# Try running queries directly
print(db.run("SELECT COUNT(*) FROM airline"))
print(db.run("SELECT * FROM airline LIMIT 5"))
print(db.run("SELECT count(*) FROM airline WHERE Gender = 'Female';"))
print(db.run("SELECT count(*) FROM airline WHERE Age > 50;"))