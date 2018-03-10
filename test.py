from app import app
import unittest

class FlaskTestCase(unittest.TestCase):

# Ensure that Flask was set up correctly

    def test_login_page(self):
    	tester = app.test_client(self)
        response = tester.get('/login', content_type='html/text')
        self.assertTrue(b'You are now logged in' in response.data)



# Ensure that login works correctly

    def test_login_page_loads(self):
    	tester = app.test_client(self)
        response = tester.get('/login', content_type='html/text')
        self.assertTrue(b'You are now logged in' in response.data)



# Ensure that login behaves correctly given the correct credentils

    def test_correct_login(self):
    	tester = app.test_client(self)
        response = tester.get(
        	'/login', data=dict(username="telan",password="12345")
        	,follow_redirects=True
        	)
        self.assertIn(b'You are now logged in', response.data)


    # Ensure that login behaves correctly given the incorrect credentils (username)

    def test_incorrect_login_usernamew(self):
    	tester = app.test_client(self)
        response = tester.get(
        	'/login', data=dict(username="sddefe",password="12345")
        	,follow_redirects=True
        	)
        self.assertIn(b'cheak your username', response.data)



    # Ensure that login behaves correctly given the incorrect credentils (password)

    def test_incorrect_login_passwordw(self):
    	tester = app.test_client(self)
        response = tester.get(
        	'/login', data=dict(username="telan",password="hfvwef")
        	,follow_redirects=True
        	)
        self.assertIn(b'incorrect password', response.data)



    # Ensure that logout works correctly

    def test_logout_loads(self):
    	tester = app.test_client(self)
        response = tester.get('/logout', content_type='html/text')
        self.assertIn(b'You are now logged out' in response.data)


    # Ensure that home page works correctly

    def test_homepage_loads(self):
    	tester = app.test_client(self)
        response = tester.get('/home', content_type='html/text')
        self.assertIn(b'This APP is an Artificial Intelligence tool that', response.data) 

    # Ensure that dashboard works correctly

    def test_dashboard_loads(self):
    	tester = app.test_client(self)
        response = tester.get('/dashboard', content_type='html/text')
        self.assertIn(b'Dashboard', response.data)

    # Ensure that add note works correctly

    def test_add_note(self):
    	tester = app.test_client(self)
        response = tester.get('/addnote', content_type='html/text')
        self.assertTrue(b'addnote' in response.data)


    # Ensure that view stock movement works correctly

    def test_stockmovement(self):
    	tester = app.test_client(self)
        response = tester.get('/stockmovement', content_type='html/text')
        self.assertTrue(b'enter a stock ticker' in response.data)


    # Ensure that tweets works correctly

    def test_tweets(self):
    	tester = app.test_client(self)
        response = tester.get('/tweets', content_type='html/text')
        self.assertTrue(b'your tweets' in response.data)

    # Ensure that live twitter sentiment works correctly

    def test_livetwittersentiment(self):
    	tester = app.test_client(self)
        response = tester.get('/livetwitter', content_type='html/text')
        self.assertTrue(b'twitter sentiment' in response.data)

    # Ensure that sentiment analysis works correctly

    def test_sentiment_analysis(self):
    	tester = app.test_client(self)
        response=tester.get('/sentiment analysis',content_type='html/text')
        self.assertIn(b'enter your stock',response.data)


    # Ensure that about page works correctly

    def test_about(self):
    	tester = app.test_client(self)
        response = tester.get('/about', content_type='html/text')
        self.assertTrue(b'about' in response.data)


    # Ensure that manage notes works correctly

    def test_manage_notes(self):
    	tester = app.test_client(self)
        response = tester.get('/managenotes', content_type='html/text')
        self.assertTrue(b'edit notes' in response.data)

    # Ensure that index page works correctly

    def test_index(self):
    	tester = app.test_client(self)
        response = tester.get('/index', content_type='html/text')
        self.assertTrue(b'AI stock predictor' in response.data)

    # Ensure that login works correctly
    def test_login_page_loads(self):
    	tester = app.test_client(self)
        response = tester.get('/login', content_type='html/text')
        self.assertTrue(b'You are now logged in' in response.data)



if __name__ == '__main__':
    unittest.main()