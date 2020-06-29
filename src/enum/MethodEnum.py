def enum(**enums):
    return type('Enum', (), enums)


MethodEnum = enum(UCB=1,
                  ODP_GP_UCB=2)
