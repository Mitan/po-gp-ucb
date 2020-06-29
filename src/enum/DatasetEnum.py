def enum(**enums):
    return type('Enum', (), enums)


DatasetEnum = enum(Simulated=1,
                   HousePrice=2,
                   Loan=3,
                   Branin=4)

