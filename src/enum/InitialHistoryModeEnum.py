def enum(**enums):
    return type('Enum', (), enums)


InitialHistoryModeEnum = enum(Random=1,
                              Deterministic=2)
