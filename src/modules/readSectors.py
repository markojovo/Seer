import csv

def get_unique_sectors(constituents_csv_file):
    sectors = set()
    with open(constituents_csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            sectors.add(row[2])  # add the sector to the set
    return list(sectors)  # return as a list

constituents_csv_file = 'constituents_csv.csv'  # replace with your file path
unique_sectors = get_unique_sectors(constituents_csv_file)

print(unique_sectors)  # print the list of unique sectors
