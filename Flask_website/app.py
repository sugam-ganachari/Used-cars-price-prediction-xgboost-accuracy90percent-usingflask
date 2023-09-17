from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('mlmodel3.pkl', 'rb'))

options = {
    0: 'Datsun Go', 1: 'Datsun Go Plus', 2: 'Datsun Redi Go', 3: 'Ford Ecosport',
    4: 'Ford FREESTYLE', 5: 'Ford New Figo', 6: 'Honda Amaze', 7: 'Honda BR-V',
    8: 'Honda Brio', 9: 'Honda City', 10: 'Honda Civic', 11: 'Honda Jazz',
    12: 'Honda WR-V', 13: 'Hyundai ALCAZAR', 14: 'Hyundai AURA', 15: 'Hyundai Creta',
    16: 'Hyundai Elite i20', 17: 'Hyundai Eon', 18: 'Hyundai GRAND I10 NIOS',
    19: 'Hyundai Grand i10', 20: 'Hyundai NEW I20', 21: 'Hyundai NEW I20 N LINE',
    22: 'Hyundai NEW SANTRO', 23: 'Hyundai New Elantra', 24: 'Hyundai Santro Xing',
    25: 'Hyundai VENUE', 26: 'Hyundai Verna', 27: 'Hyundai Xcent', 28: 'Hyundai i10',
    29: 'Hyundai i20', 30: 'Hyundai i20 Active', 31: 'Jeep Compass', 32: 'KIA CARENS',
    33: 'KIA SELTOS', 34: 'KIA SONET', 35: 'MG HECTOR', 36: 'Mahindra BOLERO NEO',
    37: 'Mahindra KUV 100 NXT', 38: 'Mahindra MARAZZO', 39: 'Mahindra Scorpio',
    40: 'Mahindra TUV300', 41: 'Mahindra Thar', 42: 'Mahindra XUV 3OO',
    43: 'Mahindra XUV500', 44: 'Maruti A Star', 45: 'Maruti Alto', 46: 'Maruti Alto 800',
    47: 'Maruti Alto K10', 48: 'Maruti BREZZA', 49: 'Maruti Baleno', 50: 'Maruti Celerio',
    51: 'Maruti Celerio X', 52: 'Maruti Ciaz', 53: 'Maruti Dzire', 54: 'Maruti Ertiga',
    55: 'Maruti IGNIS', 56: 'Maruti New Wagon-R', 57: 'Maruti Ritz', 58: 'Maruti S Cross',
    59: 'Maruti S PRESSO', 60: 'Maruti Swift', 61: 'Maruti Swift Dzire',
    62: 'Maruti Vitara Brezza', 63: 'Maruti Wagon R 1.0', 64: 'Maruti Wagon R Stingray',
    65: 'Maruti XL6', 66: 'Maruti Zen Estilo', 67: 'Nissan MAGNITE', 68: 'Nissan Micra',
    69: 'Nissan Terrano', 70: 'Renault Captur', 71: 'Renault Duster', 72: 'Renault Kiger',
    73: 'Renault Kwid', 74: 'Renault Pulse', 75: 'Renault TRIBER', 76: 'Skoda Rapid',
    77: 'Skoda SLAVIA', 78: 'Tata ALTROZ', 79: 'Tata Harrier', 80: 'Tata NEXON',
    81: 'Tata Safari', 82: 'Tata TIAGO JTP', 83: 'Tata TIGOR', 84: 'Tata Tiago',
    85: 'Tata Zest', 86: 'Toyota Etios', 87: 'Toyota Etios Liva', 88: 'Toyota Fortuner',
    89: 'Toyota Glanza', 90: 'Toyota Innova Crysta', 91: 'Toyota URBAN CRUISER',
    92: 'Toyota YARIS', 93: 'Volkswagen Ameo', 94: 'Volkswagen Jetta',
    95: 'Volkswagen Polo', 96: 'Volkswagen TAIGUN', 97: 'Volkswagen VIRTUS',
    98: 'Volkswagen Vento'
}
@app.route('/')
def index():
    return render_template('index.html',options=options)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = int(request.form['year'])
    val2 = int(request.form['car_model'])
    val3 = int(request.form['fuel'])
    val4 = int(request.form['ownership'])
    val5 = int(request.form['distance'])
    val6= int(request.form['transmission'])
    dataf= pd.DataFrame({'year':[val1],'distance_travelled':[val5],'Name_':[val2],'Transmission_':[val6],'Fuel_':[val3],'Ownership_':[val4]})
    pred=model.predict(dataf)
    print(dataf)
    return render_template('index.html',options=options, data=pred[0])


if __name__ == '__main__':
    app.run(debug=True)