import schedule
import time


def job(t):
	print ("starting from a cronjob!")
	import lstm #run NN

schedule.every().day.at("21:08").do(job,'starting to build predictions')

while True:
    schedule.run_pending()
    time.sleep(10) # wait 10 $ 
