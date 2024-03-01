import csv

class InjectionLogger:
    def __init__(self, filename):
        self.filename = filename
        self.total = 0
        self.count = 0
        self.batch_counter = 0

    def add_number(self, number):
        self.total += number
        self.count += 1

    def get_average(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def log_to_csv(self):
        
        if self.batch_counter == 0:
            with open(self.filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['batch', 'Average'])
        
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.batch_counter, self.get_average()])
            self.batch_counter += 1

# Example usage:
averager = InjectionLogger('averages.csv')
averager.add_number(10)
averager.add_number(20)
averager.add_number(30)
averager.log_to_csv()
averager.add_number(10)
averager.add_number(20)
averager.add_number(30)
averager.log_to_csv()