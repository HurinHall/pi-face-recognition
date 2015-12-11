"""Raspberry Pi Face Recognition Treasure Box
Treasure Box Script
Copyright 2013 Tony DiCola 
"""

import glob
import os
import sys
import select

import cv2

import config
import face
import time 

def is_letter_input(letter):
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False

if __name__ == '__main__':
	# Load training data into model
	print 'Loading training data...'
	model = cv2.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print 'Training data loaded!'
	# Initialize camer and box.
	camera = config.get_camera()
	personPresent = 0;

	print 'Running Hal 9000 recognition'
	print 'Press Ctrl-C to quit.'
	while True:
		# Check if capture should be made.
		# TODO: Check if button is pressed.
		#if is_letter_input('c'):
		# Check for the positive face and unlock if found.
		image = camera.read()
		# Convert image to grayscale.
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		# Get coordinates of single face in captured image.
		result = face.detect_single(image)
		if result is None:
			print 'No Face'
			time.sleep(10)
			personPresent = 0;
			continue
		x, y, w, h = result
		# Crop and resize image to face.
		crop = face.resize(face.crop(image, x, y, w, h))
		# Test face against model.
		label, confidence = model.predict(crop)
		print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(
			'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
			confidence)
		if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD and personPresent == 0:
			print 'Recognized face!'
			personPresent = 1;
			with open('/home/ubuntu/chatbotio', 'w') as file:
				file.write('Greetings Human')
			time.sleep(10);
		else:
			print 'Did not recognize face'
			time.sleep(10);
