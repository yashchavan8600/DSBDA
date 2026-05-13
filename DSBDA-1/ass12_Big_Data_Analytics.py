# Big Data Analytics
# Weather Data Analysis using Python
# STEP 1: Open Input File
file = open("sample_weather.txt", "r")

# STEP 2: Initialize Variables
temperature_sum = 0
dewpoint_sum = 0
windspeed_sum = 0

count = 0

# STEP 3: Read File Line by Line
for line in file:
    
    # Remove extra spaces and split values
    
    data = line.strip().split()

    # Convert values into float
    
    temperature = float(data[0])
    dewpoint = float(data[1])
    windspeed = float(data[2])

    # Add values to total sum
    
    temperature_sum += temperature
    dewpoint_sum += dewpoint
    windspeed_sum += windspeed

    # Increase count
    
    count += 1

# STEP 4: Calculate Averages
average_temperature = temperature_sum / count

average_dewpoint = dewpoint_sum / count

average_windspeed = windspeed_sum / count

# STEP 5: Display Results
print("Average Temperature:", average_temperature)
print("Average Dew Point:", average_dewpoint)
print("Average Wind Speed:", average_windspeed)

# STEP 6: Close File
file.close()