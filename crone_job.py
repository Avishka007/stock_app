import schedule
import time


def job(t):
	print ("starting from a cronjob!")
	import automator #to run the nn and save values to csvs

schedule.every().day.at("10:41").do(job,'starting to build predictions')

while True:
    schedule.run_pending()
    time.sleep(10) # wait 10 $ 
