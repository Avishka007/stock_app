from real_data_reader import q
import predicted_scv_slicer

#this py file is to loop NN untill all the predictions are correct

for i in q:
	if i < 0:
		import NN
	else:
		pass
