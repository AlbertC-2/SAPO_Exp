import chardet
csv_file = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.csv"

# Determine the encoding of the CSV file
with open(csv_file, 'rb') as file:
    result = chardet.detect(file.read())
    print(result['encoding'])
