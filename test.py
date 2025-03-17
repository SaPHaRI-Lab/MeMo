import csv, ast, json

with open("gestures.csv", 'r') as gestures:
    for line in gestures.readlines():
        print(ast.literal_eval(line))
        wp = ast.literal_eval(line)