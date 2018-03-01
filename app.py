from flask import Flask, render_template, flash, redirect, url_for, session, logging, request 
#from data import Tweets
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps

app = Flask(__name__)

#config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

#init MYSQL
mysql = MySQL(app)


#Tweets = Tweets ()

#Cheak if user logged in
def is_logged_in(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args, **kwargs)
		else:
			flash('Unauthorized acess, Please login', 'danger')
			return redirect(url_for('login'))
	return wrap


#mul ma page 1
@app.route('/')
def home():
	return render_template ('home.html')

#index page
@app.route('/index')
def index():
	return render_template ('index.html')

#about page
@app.route('/about')
def about():
	return render_template ('about.html')

#then tweets/ then articles
@app.route('/articles')
@is_logged_in
def tweets():
	#Create cursor
	cur = mysql.connection.cursor()

	#get notes
	resault = cur.execute("SELECT * FROM articles")

	articles = cur.fetchall()

	if resault > 0:
		return render_template('articles.html', articles=articles)
	else:
		msg = 'No Notes Found!'
		return render_template('articles.html', msg=msg)

	#close connection
	cur.close()


#single tweet
@app.route('/tweets/<string:id>/')
def tweetsid(id):
	#Create cursor
	cur = mysql.connection.cursor()

	#get notes
	resault = cur.execute("SELECT * FROM articles WHERE id = %s",[id])

	tweet = cur.fetchone()

	return render_template ('tweet.html', tweet=tweet)




#Register form class
class RegisterForm(Form):
	name = StringField ('Name', [validators.Length(min=1, max=50)])
	username = StringField ('Username', [validators.Length(min=4, max=25)])
	email = StringField ('Email', [validators.Length(min=6, max=50)])
	password = PasswordField('Password',[
		validators.DataRequired(),
		validators.EqualTo('confirm', message = 'Passwords do not match')
		])
	confirm = PasswordField('Confirm Password')

#User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
	form = RegisterForm(request.form)
	if request.method == 'POST' and form.validate():
		name = form.name.data
		email = form.email.data
		username = form.username.data
		password = sha256_crypt.encrypt(str(form.password.data))


		#Create cursor
		cur = mysql.connection.cursor()

		# Execute query
		cur.execute("INSERT INTO users(name, email, username, password) VALUES (%s, %s, %s, %s)", (name, email, username, password))

		#Commit to DB
		mysql.connection.commit()

		#Close connection
		cur.close()

		flash('You are sucessfuly registered !', 'success')

		return redirect (url_for('index'))
	return render_template('register.html',form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'POST':
		#1stly get from fields
		username = request.form['username']
		password_candidate = request.form['password']

		#create cursor
		cur = mysql.connection.cursor()

		#Get by username
		resault = cur.execute("SELECT * FROM users WHERE username = %s", [username])

		#cheak the resaults
		if resault > 0:
			#get stored hash
			data = cur.fetchone()
			password = data['password']

			#Compare Passwords
			if sha256_crypt.verify(password_candidate, password):
				#app.logger.info("PASSWORED MATCHED")
				#Passed
				session['logged_in'] = True
				session['username'] = username

				flash('You are now logged in', 'success')
				return redirect(url_for('dashboard'))

			else:
				error = 'Please cheak your password'
				return render_template('login.html', error=error)

			#Close the connection
			cur.close()

		else:
			error = 'Username not Found'
			return render_template('login.html', error=error)

	return render_template('login.html')



#Logout
@app.route('/logout')
@is_logged_in
def logout():
	#kill sessions
	session.clear()
	flash('You are now logged out', 'success')
	return redirect(url_for('login'))


#Dashboard
@app.route('/dashboard')
@is_logged_in
def dashboard():
	return render_template('dashboard.html')

#note
@app.route('/note')
@is_logged_in
def note():
	#Create cursor
	cur = mysql.connection.cursor()

	#get notes
	resault = cur.execute("SELECT * FROM articles")

	articles = cur.fetchall()

	if resault > 0:
		return render_template('note.html', articles=articles)
	else:
		msg = 'No Notes Found!'
		return render_template('note.html', msg=msg)

	#close connection
	cur.close()


#Article form class
class ArticleForm(Form):
	title = StringField ('Title', [validators.Length(min=1, max=20)])
	body = StringField ('Body', [validators.Length(min=10)])

#Add article(note)
@app.route('/add_article', methods=['GET', 'POST'])
@is_logged_in
def add_article():
	form = ArticleForm(request.form)
	if request.method == 'POST' and form.validate():
		title = form.title.data
		body = form.body.data

		#create cursor
		cur = mysql.connection.cursor()

		#Execute
		cur.execute("INSERT INTO articles (title, body, author) VALUES( %s, %s, %s)",(title, body, session['username']))

		#Commit to DB
		mysql.connection.commit()

		#close connection
		cur.close()

		flash('Note Added', 'success')

		return redirect(url_for('dashboard'))

	return render_template('add_article.html', form=form)


#edit article(note)
@app.route('/edit_article/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_article(id):
	#create cursor
	cur = mysql.connection.cursor()

	#Get article by id
	resault = cur.execute("SELECT * FROM articles WHERE id = %s",[id])

	article = cur.fetchone()

	#get form
	form = ArticleForm(request.form)

	#populate article form fields
	form.title.data = article['title']
	form.body.data = article['body']


	if request.method == 'POST' and form.validate():
		title = request.form['title']
		body = request.form['body']

		#create cursor
		cur = mysql.connection.cursor()

		#Execute
		cur.execute("UPDATE articles SET title=%s, body=%s WHERE id = %s", (title, body, id))

		#Commit to DB
		mysql.connection.commit()

		#close connection
		cur.close()

		flash('Note Modified', 'success')

		return redirect(url_for('dashboard'))

	return render_template('edit_article.html', form=form)


#Delete note
@app.route('/delete_article/<string:id>', methods = ['POST'])
@is_logged_in
def delete_article(id):
	#create cursor
	cur = mysql.connection.cursor()

	#Execute
	cur.execute("DELETE FROM articles WHERE id =%s",[id])

	#Commit to DB
	mysql.connection.commit()

	#close connection
	cur.close()

	flash('Note Deleted', 'success')

	return redirect(url_for('note'))




if __name__ == '__main__':
	app.secret_key = 'secret123'
	app.run(debug=True)
