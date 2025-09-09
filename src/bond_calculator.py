def calculate_bond_return(principal, rate, months, compounding=False):
    if compounding:
        return principal * ((1 + (rate / (100 * 12))) ** months)
    return principal + (principal * rate * (months / 12) / 100)