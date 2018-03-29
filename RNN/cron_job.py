#this py file will grab stock data and run NN every day right after NYSE announses their closing prices.

import schedule
import time


def job(t):
	print ("starting from a cronjob!")
	import web_scraper #to update all stock values
	import automator #to run the nn and save values to csvs

schedule.every().day.at("10:41").do(job,'starting to build predictions')

while True:
    schedule.run_pending()
    time.sleep(10) # wait 10 $ 
