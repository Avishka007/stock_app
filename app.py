from flask import Flask, render_template
from data import Tweets


app = Flask(__name__)

Tweets = Tweets ()

@app.route('/')
def home():
	return render_template ('home.html')

@app.route('/index')
def index():
	return render_template ('index.html')

@app.route('/about')
def about():
	return render_template ('about.html')

@app.route('/tweets')
def tweets():
	return render_template ('tweets.html', tweets = Tweets)

@app.route('/tweets/<string:id>/')
def tweetsid(id):
	return render_template ('tweet.html', id=id)

if __name__ == '__main__':
	app.run(debug=True)
