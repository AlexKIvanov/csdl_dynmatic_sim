import csv

csvDir = '/home/alexander/Documents/OptResults/DynamicsResults/'

def read_in_csvs(listNames : list):

    allResults = {}
    for name in listNames: 
        with open(csvDir + name + '.csv', mode='r',newline='\n') as myFile:
            reader = csv.DictReader(myFile)

            allResults[name] = dict()
            for row in reader:
                for column, value in row.items():  
                    if value == '':
                        pass
                    else:
                        allResults[name].setdefault(column, []).append(value)


    return allResults