from flask import Flask
import yfinance as yf 

app = Flask(__name__)

@app.route("/")
def home():
    return 'Welcome!!!'

@app.route("/finance/<ticker>")
def finance(ticker):
    yfObject = yf.Ticker(ticker)
    financial = yfObject.financials 
    return financial.to_json()

@app.route("/hello/<name>")
def hello_world(name):
    return 'Hello %s!' % name

if __name__ == '__main__':
    app.run()
