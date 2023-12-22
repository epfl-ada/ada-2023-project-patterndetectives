import pandas as pd
import numpy as np

def get_inflation(df: pd.DataFrame):
    columns_inf = ['year', 'amount','inflation rate']
    inflation = pd.read_table('../data/inflation_data.csv', header=None, names=columns_inf,sep=',')
    inflation = inflation.drop(index=0)

    #From https://www.officialdata.org/us/inflation/1888?amount=1

    value_in_2023 = [32.39,33.44,33.81,33.81,33.81,34.19,35.78,36.63,36.63,37.07,
                       37.07,37.07,36.63,36.20,35.78,34.96,34.57,34.96,34.19,32.73,
                       33.44,33.81,32.39,32.39,31.72,31.08,30.77,30.46,28.23,24.04,
                       20.38,17.78,15.38,17.19,18.31,17.99,17.99,17.58,17.38,17.68,
                       17.99,17.99,18.42,20.24,22.46,23.67,22.96,22.46,22.13,21.37,
                       21.82,22.13,21.98,20.93,18.88,17.78,17.48,17.09,15.78,13.80,
                       12.77,12.93,12.77,11.83,11.61,11.52,11.44,11.48,11.31,10.95,
                       10.65,10.57,10.39,10.29,10.19,10.05,9.92,9.77,9.50,9.21,8.84,
                       8.38,7.93,7.60,7.36,6.93,6.24,5.72,5.41,5.08,4.72,4.24,3.73,
                       3.38,3.19,3.09,2.96,2.86,2.81,2.71,2.60,2.48,2.35,2.26,2.19,
                       2.13,2.08,2.02,1.96,1.92,1.89,1.85,1.79,1.74,1.71,1.67,1.63,
                       1.58,1.53,1.48,1.43,1.43,1.41,1.37,1.34,1.32,1.30,1.30,1.28,
                       1.26,1.22,1.20,1.19,1.14,1.05,1]

    inflation["Inflation Factor for 2023"] = value_in_2023
    inflation["year"] = inflation["year"].astype(float)

    df['Inflation Factor for 2023'] = df['Movie_release'].map(inflation.set_index('year')['Inflation Factor for 2023'])
    df['2023 valued revenue'] = df['Movie_revenue'] * df['Inflation Factor for 2023']

    df = df.sort_values(by=['2023 valued revenue'],ascending = False)

    return df

def round_down_to_nearest_05(number):
    return np.floor(number / 0.05) * 0.05

def interpolate_color(ratio,start_rgb,end_rgb):

    # Linearly interpolate each color component
    r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
    g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
    b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio

    return (r/255, g/255, b/255)

# Function to transform x to y and create a tuple
def transform(x):
    if x >= 0.5:
        start_rgb = (112,85,137)
        end_rgb = (229, 83, 159)
        y = (x - 0.5) * 2
        return interpolate_color(y,start_rgb,end_rgb)
    else:
        y = np.abs((x - 0.5) * 2)
        start_rgb = (57,35,35)
        end_rgb = (112,85,137)        
        return interpolate_color(y,start_rgb,end_rgb)
    
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# Function to compute average color
def average_color(colors):
    avg = np.mean(colors, axis=0)
    return rgb_to_hex(avg)