import csv
from datetime import datetime
import numpy.ma as ma
# Title: Making the data ready for the models
# Trying to Make the data into one file with the right dates
# Author: Arnav singh

# Lists to store dates and values
dates = []
values = []
# Open the file
with (open('OMNI2_H0_MRG1HR_729923.txt', 'r') as file):
    # Flag to indicate if we should start reading data
    start_reading = False
    # Iterate over each line in the file
    for line in file:
        # Check if the line starts with the target date and time, skips the header
        if line.startswith('01-12-1983 00:30:00.000'):
            # Set the flag to True to start reading data
            start_reading = True
        # If the flag is True, process the data
        if start_reading:
            # Split the line by whitespace
            data = line.split()
            # Ensure it's not an empty line and has enough elements
            if data and len(data) >= 3:  # Ensure at least three elements exist
                # Try to convert the value to an integer
                try:
                    value = int(data[2])  # Convert value to integer
                except ValueError:
                    # Skip this line if the value is not a valid integer
                    continue
                # Assuming you want the second and third columns
                date_str = data[0] + ' ' + data[1]
                # Convert date string to datetime object
                date = datetime.strptime(date_str, '%d-%m-%Y %H:%M:%S.%f')
                # Append date and value to lists
                dates.append(date)
                values.append(value)

datevalue_pairs = [(date, value) for date, value in zip(dates, values) if value < 5000]
# Unzip the filtered pairs
date_filtered, value_filtered = zip(*datevalue_pairs)
dates = date_filtered
values = value_filtered


with open("test2.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Date(UTC)", "AE"])
    for i in range(len(values)):
        writer.writerow([dates[i], values[i]])
