import pandas as pd

df = pd.read_csv('../Forecasting/Forecasting/EnergyConsuption/pjm_hourly_est.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')

print('Data range:', df['Datetime'].min(), 'to', df['Datetime'].max())
print('\nColumn data ranges:')

cols = ['AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE', 'NI', 'PJME', 'PJMW', 'PJM_Load']
for col in cols:
    non_null = df[col].notnull().sum()
    if non_null > 0:
        first = df[df[col].notnull()]['Datetime'].min()
        last = df[df[col].notnull()]['Datetime'].max()
        print(f'{col}: {non_null} values from {first} to {last}')
