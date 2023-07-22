import RPi.GPIO as GPIO
import time
import cv2
import tensorflow as tf
import numpy as np

# Define the GPIO pins for the ultrasonic sensors
TRIG_1 = 4
ECHO_1 = 5
TRIG_2 = 6
ECHO_2 = 7

# Define the GPIO pins for the servo motor and Arduino
SERVO_PIN = 17
ARDUINO_PIN = 21

# Initialize the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_1, GPIO.OUT)
GPIO.setup(ECHO_1, GPIO.IN)
GPIO.setup(TRIG_2, GPIO.OUT)
GPIO.setup(ECHO_2, GPIO.IN)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(ARDUINO_PIN, GPIO.OUT)

# Load the bottle classifier model
model = tf.keras.models.load_model('bottle_classifier.h5')

# Create a function to classify the bottle
def bottle_classifier(image):
    # Preprocess the image
    image = cv2.resize(image, dsize=(64, 64))
    image = image.flatten()
    image_array = np.array(image)

    # Classify the bottle
    prediction = model.predict(image_array)

    # Convert the prediction to a bottle type
    def output_converter(model_output):

        import numpy as np

        output = model_output

        # assume that 'output' is a numpy array of shape (n, 2)
        output_labels = ['0.5 L', '1 L']
        predictions = np.argmax(output, axis=1)
        predicted_labels = [output_labels[p] for p in predictions]

        return predicted_labels[0]

    predicted_labels = output_converter(prediction)

    return predicted_labels[0]

# Main loop
while True:

    # Capture the image
    capture = cv2.VideoCapture(0)
    success, image = capture.read()

    # Check the distance to the two objects
    distance_1 = calculate_distance(measure_echo_pulse(ECHO_1))
    distance_2 = calculate_distance(measure_echo_pulse(ECHO_2))

    # If the distance to the first object is less than 15 cm and the distance to the second object is greater than 15 cm, classify the bottle, send the appropriate signal to the Arduino, and print the bottle type
  bottle_type = bottle_classifier(image)  
  if distance_1 < 15 and distance_2 > 15 and bottle_type == '0.5L':
    GPIO.output(ARDUINO_PIN, GPIO.HIGH)
    print('The bottle is a 0.5L bottle.')
  else:
    GPIO.output(ARDUINO_PIN, GPIO.LOW)
    print('The bottle is a 1L bottle.')

time.sleep(10)

