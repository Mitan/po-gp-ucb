def enum(**enums):
    return type('Enum', (), enums)


AcquisitionEnum = enum(UCB=1)
