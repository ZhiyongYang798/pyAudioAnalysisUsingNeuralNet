import sys, os, alsaaudio, time, audioop, numpy, glob,  scipy, subprocess, wave, cPickle, threading, shutil
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft
import audioFeatureExtraction as aF	
import audioTrainTest as aT
import audioSegmentation as aS
from scipy.fftpack import fft
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')

Fs = 16000

def recordAudioSegments(RecordPath, BLOCKSIZE):	
	# This function is used for recording audio segments (until ctr+c is pressed)
	# ARGUMENTS:
	# - RecordPath:		the path where the wav segments will be stored
	# - BLOCKSIZE:		segment recording size (in seconds)
	# 
	# NOTE: filenames are based on clock() value
	
	print "Press Ctr+C to stop recording"
	RecordPath += os.sep
	d = os.path.dirname(RecordPath)
	if os.path.exists(d) and RecordPath!=".":
		shutil.rmtree(RecordPath)	
	os.makedirs(RecordPath)	

	inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)
	inp.setchannels(1)
	inp.setrate(Fs)
	inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
	inp.setperiodsize(512)
	midTermBufferSize = int(Fs*BLOCKSIZE)
	midTermBuffer = []
	curWindow = []
	elapsedTime = "%08.3f" % (time.time())
	while 1:
			l,data = inp.read()		   
		    	if l:
				for i in range(len(data)/2):
					curWindow.append(audioop.getsample(data, 2, i))
		
				if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
					samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
				else:
					samplesToCopyToMidBuffer = len(curWindow)

				midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];
				del(curWindow[0:samplesToCopyToMidBuffer])
			

			if len(midTermBuffer) == midTermBufferSize:
				# allData = allData + midTermBuffer				
				curWavFileName = RecordPath + os.sep + str(elapsedTime) + ".wav"				
				midTermBufferArray = numpy.int16(midTermBuffer)
				wavfile.write(curWavFileName, Fs, midTermBufferArray)
				print "AUDIO  OUTPUT: Saved " + curWavFileName
				midTermBuffer = []
				elapsedTime = "%08.3f" % (time.time())	

def neuralNetClassidication(duration, midTermBufferSizeSec, modelName):
	n_dim = 68
	n_classes = 2
	n_hidden_units_one = 280 
	n_hidden_units_two = 300
	sd = 1 / numpy.sqrt(n_dim)
	learning_rate = 0.01

	X = tf.placeholder(tf.float32,[None,n_dim])
	Y = tf.placeholder(tf.float32,[None,n_classes])

	W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd), name = "W_1")
	b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd), name = "b_1")
	h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


	W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd), name = "W_2")
	b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd), name = "b_2")
	h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


	W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name = "W")
	b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name = "b")
	y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

	saver = tf.train.Saver()

	cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1])) 
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	cost_history = numpy.empty(shape=[1],dtype=float)
	y_true, y_pred = None, None

	inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK)
	inp.setchannels(1)
	inp.setrate(Fs)
	inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
	inp.setperiodsize(512)
	midTermBufferSize = int(midTermBufferSizeSec * Fs)
	allData = []
	midTermBuffer = []
	curWindow = []
	count = 0


	with tf.Session() as sess:
	    # sess.run(init)

	    saver.restore(sess, modelName)
	    while len(allData)<duration*Fs:
	# Read data from device
			l,data = inp.read()
		    	if l:
				for i in range(l):
					curWindow.append(audioop.getsample(data, 2, i))		
				if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
					samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
				else:
					samplesToCopyToMidBuffer = len(curWindow)
				midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];
				del(curWindow[0:samplesToCopyToMidBuffer])
			if len(midTermBuffer) == midTermBufferSize:
				count += 1						
				[mtFeatures, stFeatures] = aF.mtFeatureExtraction(midTermBuffer, Fs, 2.0*Fs, 2.0*Fs, 0.020*Fs, 0.020*Fs)
				features = numpy.array([mtFeatures[:,0]])	
				y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: features})
				if y_pred[0] == 0:
					print "Class A"
				else:
					print "Class B"
				allData = allData + midTermBuffer

				plt.clf()
				plt.plot(midTermBuffer)
				plt.show(block = False)
				plt.draw()
				midTermBuffer = []
	
def recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType):
	'''
	recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType)

	This function is used to record and analyze audio segments, in a fix window basis.

	ARGUMENTS: 
	- duration			total recording duration
	- outputWavFile			path of the output WAV file
	- midTermBufferSizeSec		(fix)segment length in seconds
	- modelName			classification model name
	- modelType			classification model type

	'''
	if modelType == 'neuralnet':
		neuralNetClassidication(duration, midTermBufferSizeSec, modelName)
	else:

		if modelType=='svm':
			[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
		elif modelType=='knn':
			[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel(modelName)
		else:
			Classifier = None

		inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK)
		inp.setchannels(1)
		inp.setrate(Fs)
		inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
		inp.setperiodsize(512)
		midTermBufferSize = int(midTermBufferSizeSec * Fs)
		allData = []
		midTermBuffer = []
		curWindow = []
		count = 0
# a sequence of samples
# process a sequence 
# speed
# emergency vehicle detection what have they done? emergency vehicle classification patents
# plot features!!!
# patents extracted!
# latex literature review 
# writing a paper
# 
		while len(allData)<duration*Fs:
			# Read data from device
			l,data = inp.read()
		    	if l:
				for i in range(l):
					curWindow.append(audioop.getsample(data, 2, i))		
				if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
					samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
				else:
					samplesToCopyToMidBuffer = len(curWindow)
				midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];
				del(curWindow[0:samplesToCopyToMidBuffer])
			if len(midTermBuffer) == midTermBufferSize:
				count += 1						
				if Classifier!=None:
					[mtFeatures, stFeatures] = aF.mtFeatureExtraction(midTermBuffer, Fs, 2.0*Fs, 2.0*Fs, 0.020*Fs, 0.020*Fs)
					curFV = (mtFeatures[:,0] - MEAN) / STD;
					[result, P] = aT.classifierWrapper(Classifier, modelType, curFV)
					print classNames[int(result)]
				allData = allData + midTermBuffer

				plt.clf()
				plt.plot(midTermBuffer)
				plt.show(block = False)
				plt.draw()


				midTermBuffer = []

		allDataArray = numpy.int16(allData)
		wavfile.write(outputWavFile, Fs, allDataArray)

def main(argv):
	if argv[1] == '-recordSegments':		# record input
		if (len(argv)==4): 			# record segments (until ctrl+c pressed)
			recordAudioSegments(argv[2], float(argv[3]))
		else:
			print "Error.\nSyntax: " + argv[0] + " -recordSegments <recordingPath> <segmentDuration>"

	if argv[1] == '-recordAndClassifySegments':	# record input
		if (len(argv)==6):			# recording + audio analysis
			duration = int(argv[2])
			outputWavFile = argv[3]
			modelName = argv[4]
			modelType = argv[5]
			if modelType not in ["svm", "knn", "neuralnet"]:
				raise Exception("ModelType has to be either svm or knn or neuralnet!")
			if modelType == "neuralnet": #improve!!!
				recordAnalyzeAudio(duration, outputWavFile, 2.0, modelName, modelType)
			else:
				if not os.path.isfile(modelName):
					raise Exception("Input modelName not found!")
				recordAnalyzeAudio(duration, outputWavFile, 2.0, modelName, modelType)
		else:
			print "Error.\nSyntax: " + argv[0] + " -recordAndClassifySegments <duration> <outputWafFile> <modelName> <modelType>"
	
if __name__ == '__main__':
	main(sys.argv)
